"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import sys
import pickle
import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import parse_args, check_and_setup, copy_to_nfs


if __name__ == '__main__':
    args = parse_args('Prepare Shakespeare char-level dataset')
    
    # Check NFS first, then local - if files exist, we're done
    result = check_and_setup('shakespeare_char', args.force, ['train.bin', 'val.bin', 'meta.pkl'])
    if result is None:
        exit(0)
    
    # Files don't exist - need to process
    local_dir, nfs_dir = result
    
    input_file_path = os.path.join(local_dir, 'input.txt')
    if not os.path.exists(input_file_path) or args.force:
        print("Downloading...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    
    with open(input_file_path, 'r') as f:
        data = f.read()
    
    print("Creating vocabulary...")
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"vocab size: {vocab_size}, chars: {''.join(chars)}")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train: {len(train_ids):,} tokens, val: {len(val_ids):,} tokens")
    
    print("Writing bin files...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(local_dir, 'train.bin'))
    val_ids.tofile(os.path.join(local_dir, 'val.bin'))
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(local_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    if nfs_dir:
        print("Copying to NFS...")
        copy_to_nfs(local_dir, nfs_dir, ['train.bin', 'val.bin', 'meta.pkl'])
    
    print(f"Done! Files in {local_dir}")
