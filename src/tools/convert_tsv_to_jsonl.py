from tqdm import tqdm
import json
import argparse
import os

def main(args):
    
    fout = open(args.path_jsonl, 'w')
    with open(args.path_tsv, 'r') as f:
        for line in tqdm(f):
            title, contents, id = line.strip().split('\t')
            fout.write(json.dumps({
                "id": f"{title}#{id}", 
                "title": title, 
                "contents": content
            }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_tsv", type=str)
    parser.add_argument("--path_jsonl", type=str)
    args = parser.parse_args()

    main(args)
