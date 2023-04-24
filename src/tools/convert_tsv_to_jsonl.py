import os
import re
from tqdm import tqdm
import json
import argparse

def main(args):
    
    fout = open(args.path_jsonl, 'w')
    with open(args.path_tsv, 'r') as f:
        next(f)
        for line in tqdm(f):
            line = re.sub('"', '', line)
            id_, text, title = line.strip().split('\t')
            title_topic = title.split(' [SEP] ')[0]
            if args.add_title:
                fout.write(json.dumps({
                    "id": f"{title_topic}#{id_}", 
                    "contents": f"{title_topic} {text}"
                }, ensure_ascii=False)+'\n')
            else:
                fout.write(json.dumps({
                    "id": f"{title_topic}#{id_}", 
                    "title": title_topic, 
                    "contents": text
                }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_tsv", type=str)
    parser.add_argument("--path_jsonl", type=str)
    parser.add_argument("--add_title", default=False, action='store_true')
    args = parser.parse_args()

    main(args)
