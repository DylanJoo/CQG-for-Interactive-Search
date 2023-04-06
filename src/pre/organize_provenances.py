import re
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import unicodedata

def normalize(text):
    return unicodedata.normalize("NFD", text)

def load_collections(path):
    collections = {}
    print("Load collections...")
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            collections[data['id']] = normalize(data['contents'])
    print("Done!")
    return collections

def overlapped_provenances(plist_q, plist_cq, N):
    """
    Parameters
    ----------
    N: int
        Number of provenances should be considered, 
        including list of q and list of cq

    Returns
    -------
    serp: List 
        The list of overlapped provenances of two lists.
    """
    # the overlapped
    serp = [docid for docid in plist_q if docid in plist_cq]
    doc_q = [docid for docid in plist_q if docid not in serp]
    doc_cq = [docid for docid in plist_cq if docid not in serp]

    n = len(serp)
    if n < N:
        offset = math.ceil((N-n)/2)
        serp += doc_cq[:offset]
        serp += doc_q[:offset]

    # [NOTE] Move the randomly sampled orders to collator

    return serp[:N]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--questions_with_provenances", type=str)
    parser.add_argument("--collections", type=str)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    parser.add_argument("--N", default=20, type=int)
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
            if 'cq_serp' in data.keys(): # i.e., ClariQ
                serp_list1, serp_scores1 = data.pop('cq_serp')
            elif 'q_serp' in data.keys(): # i.e., CQA
                serp_list1, serp_scores1 = data.pop('a_serp')
            else: # i.e., ClariQ
                serp_list1, serp_scores1 = serp_list0, q_serp_scores 

            serp = overlapped_provenances(serp_list0, serp_list1, args.N)

            data.update({
                "titles": [docid.split("#")[0] for docid in serp],
                "provenances": [collections[docid] for docid in serp],
            })
            fout.write(json.dumps(data, ensure_ascii=False)+'\n')
