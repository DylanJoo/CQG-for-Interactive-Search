import os
import re
from tqdm import tqdm
import json
import argparse
import numpy as np
from datasets import load_dataset

def main(args):
    
    fout = open(args.path_txt, 'w')
    with open(args.path_jsonl, 'r') as f:
        for line in tqdm(f):
            data_dict = json.loads(line.strip())
            print("question\t", data_dict['question'])
            print("clarification\t", data_dict['prediction'])
            print(f"scores: (mean) {np.mean(data_dict['score_mu'])} (std) {np.mean(data_dict['score_sigma'])}")

            fout.write(f"{data_dict['question']} --> {data_dict['prediction']}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_jsonl", type=str)
    parser.add_argument("--path_txt", type=str)
    args = parser.parse_args()

    main(args)
