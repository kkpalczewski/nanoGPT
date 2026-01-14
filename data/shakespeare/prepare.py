import os
import sys
import requests
import tiktoken
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import parse_args, check_and_setup, copy_to_nfs


if __name__ == '__main__':
    args = parse_args('Prepare Shakespeare dataset')
    
    # Check NFS first, then local - if files exist, we're done
    result = check_and_setup('shakespeare', args.force, ['train.bin', 'val.bin'])
    if result is None:
        exit(0)
    
    # Files don't exist - need to process
    local_dir, nfs_dir = result
    
    input_file_path = os.path.join(local_dir, 'input.txt')
    if not os.path.exists(input_file_path) or args.force:
        print("Downloading...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    print("Encoding...")
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train: {len(train_ids):,} tokens, val: {len(val_ids):,} tokens")
    
    print("Writing bin files...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(local_dir, 'train.bin'))
    val_ids.tofile(os.path.join(local_dir, 'val.bin'))
    
    if nfs_dir:
        print("Copying to NFS...")
        copy_to_nfs(local_dir, nfs_dir, ['train.bin', 'val.bin'])
    
    print(f"Done! Files in {local_dir}")
