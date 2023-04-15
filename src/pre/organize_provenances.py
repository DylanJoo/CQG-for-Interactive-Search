import re
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import unicodedata

def load_collections(path):
    def normalize(text):
        return unicodedata.normalize("NFD", text)
    collections = {}
    print("Load collections...")
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            collections[data['id']] = normalize(data['contents'])
    print("Done!")
    return collections

def overlapped_provenances(plist_q, plist_qcq, N):
    """ Organize the overlapped passages as the final provencens """
    # the overlapped
    serp = [docid for docid in plist_q if docid in plist_qcq]
    doc_q = [docid for docid in plist_q if docid not in serp]
    doc_qcq = [docid for docid in plist_qcq if docid not in serp]

    n = len(serp)
    if n < N:
        offset = math.ceil((N-n)/2)
        serp += doc_qcq[:offset]
        serp += doc_q[:offset]

    # [NOTE] Move the randomly sampled orders to collator
    return serp[:N]

def exclusive_provenances(plist_q, plist_qa, N):
    """ Organize the passages with exclusion of the QA retrieved.
    as the final provencens """
    # the overlapped
    serp = [docid for docid in plist_q if docid not in plist_qa]
    doc_q = [docid for docid in plist_q if docid not in serp]
    # doc_qa = [docid for docid in plist_qa if docid not in serp]
    return (serp+doc_q)[:N]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--questions_with_provenances", type=str)
    parser.add_argument("--collections", type=str)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    parser.add_argument("--N", default=20, type=int)
    parser.add_argument("--overlapped", default=False, action='store_true')
    parser.add_argument("--exclusive", default=False, action='store_true')
    args = parser.parse_args()

    # Load collections for docid
    collections = load_collections(args.collections)

    # Distributing passages to each query
    with open(args.questions_with_provenances, 'r') as fin, \
         open(args.output, 'w') as fout:
        for line in tqdm(fin):
            data = json.loads(line.strip())

            ## Overlapped provenance
            serp_list0, q_serp_scores = data.pop('q_serp')
            serp_list1, serp_scores1 = data.pop('ref_serp', None)
            assert (serp_list1 is not None) == (args.overlapped and args.exclusive), 'Cannot find two SERP in the data dict.'

            if args.overlapped:
                serp = overlapped_provenances(serp_list0, serp_list1, args.N)
            elif args.exclusive:
                serp = exclusive_provenances(serp_list0, serp_list1, args.N)
            else:
                serp = serp_list0[:args.N]


            data.update({
                "titles": [docid.split("#")[0] for docid in serp],
                "provenances": [collections[docid] for docid in serp],
            })
            fout.write(json.dumps(data, ensure_ascii=False)+'\n')
