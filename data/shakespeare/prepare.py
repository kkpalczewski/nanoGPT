import os
import sys
import requests
import tiktoken
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import parse_args, setup_output_dir


if __name__ == '__main__':
    args = parse_args('Prepare Shakespeare dataset')
    output_dir = setup_output_dir('shakespeare', args.force)
    
    if output_dir is None:
        exit(0)

    # download the tiny shakespeare dataset
    input_file_path = os.path.join(output_dir, 'input.txt')
    if not os.path.exists(input_file_path) or args.force:
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print("Downloading Shakespeare dataset...")
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    print("Done!")

    # train.bin has 301,966 tokens
    # val.bin has 36,059 tokens
