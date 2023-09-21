import re
import math
import json
import argparse
import pandas as pd
import random
from tqdm import tqdm
from datasets import load_dataset
from loader import load_collections

def overlapped_provenances(list1, list2, N):
    serp = [docid for docid in list1 if docid in list2]
    docs1 = [docid for docid in list1 if docid not in serp]
    docs2 = [docid for docid in list2 if docid not in serp]
    n = len(serp)
    # add doc in each list if serp is less than N
    if n < N:
        offset = math.ceil((N-n)/2)
        serp += docs1[:offset]
        serp += docs2[:offset]
    return serp[:N]

def exclusive_provenances(list1, list2, N):
    serp = [docid for docid in list1 if docid not in list2]
    # add doc in list1 if serp is less than N
    docs1 = [docid for docid in list1 if docid not in serp]
    return (serp+docs1)[:N]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--collections", type=str)
    parser.add_argument("--N", default=20, type=int)
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--overlapped", default=False, action='store_true')
    parser.add_argument("--exclusive", default=False, action='store_true')
    parser.add_argument("--random", default=False, action='store_true')
    args = parser.parse_args()

    # Load collections for docid
    collections = load_collections(args.collections)

    # Distributing passages to each query
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:

        for line in tqdm(fin):
            data = json.loads(line.strip())
            serp_base = data.pop('q_serp')
            serp_ref = data.pop('ref_serp', [[""], [0]])

            assert (args.overlapped or args.exclusive), \
                    'Please specify at least one strategy.'

            if args.overlapped:
                serp = overlapped_provenances(
                        serp_base[0], serp_ref[0], args.N
                )
            elif args.exclusive:
                serp = exclusive_provenances(
                        serp_base[0], serp_ref[0], args.N
                )
            # [ablation]
            elif args.random:
                serp = [serp_base[0][i] for i in random.choices(
                    range(0, len(serp_base[0])-1), k=args.N)]

            # change names
            if 'cqg' in args.output:
                data['target'] = data.pop('c_question')
            if 'qa' in args.output:
                data['target'] = data.pop('answer')
            data.update({
                "provenances": [collections[docid] for docid in serp],
            })
            fout.write(json.dumps(data, ensure_ascii=False)+'\n')
