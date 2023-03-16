import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def load_collections(path):
    collections = {}
    print("Load collections...")
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            collections[data['id']] = data['contents']
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
    doc_cq = [docid for docid in plist_q if docid not in serp]
    n = len(serp)
    if n < N:
        offset = math.ceil(N-n) // 2
        serp += doc_q[:offset]
        serp += doc_cq[:offset]

    return serp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--clariq_provenances", type=str)
    parser.add_argument("--collections", type=str)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    parser.add_argument("--N", default=20, type=int)
    args = parser.parse_args()

    # Load collections for docid
    collections = load_collections(args.collections)

    # Distributing passages to each query
    with open(args.clariq_provenances, 'r') as fin, \
         open(args.output, 'w') as fout:
        for line in tqdm(fin):
            data = json.loads(line.strip())

            ## Overlapped provenance
            q_serp_list, q_serp_scores = data['q_serp']
            cq_serp_list, cq_serp_scores = data['cq_serp']
            serp = overlapped_provenances(q_serp_list, cq_serp_list, args.N)

            fout.write(json.dumps({
                "question": data['question'],
                "c_question": data['c_question'],
                "provenance": [collections[docid] for docid in serp],
                "c_need": data['c_need'],
            }, ensure_ascii=False)+'\n')
            # for rank, docid in enumerate(serp):
            #     fout.write(json.dumps({
            #         "question": data['question'],
            #         "c_question": data['c_question'],
            #         "provenance": collections[docid],
            #         "c_need": data['c_need'],
            #         "provenance_rank": rank
            #     }, ensure_ascii=False)+'\n')




