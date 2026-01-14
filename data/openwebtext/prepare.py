# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import sys
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import parse_args, check_and_setup, copy_to_nfs

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")


if __name__ == '__main__':
    args = parse_args('Prepare OpenWebText dataset')
    
    # Check NFS first, then local - if files exist, we're done
    result = check_and_setup('openwebtext', args.force, ['train.bin', 'val.bin'])
    if result is None:
        exit(0)
    
    # Files don't exist - need to process
    local_dir, nfs_dir = result
    
    print("Downloading dataset...")
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}
    
    print("Tokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing",
        num_proc=num_proc,
    )
    
    print("Writing bin files...")
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(local_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'{split}.bin'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    if nfs_dir:
        print("Copying to NFS...")
        copy_to_nfs(local_dir, nfs_dir, ['train.bin', 'val.bin'])
    
    print(f"Done! Files in {local_dir}")
