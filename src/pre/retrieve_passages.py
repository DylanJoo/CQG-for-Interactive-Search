import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher

def sparse_retrieve(queries, args):
    """ This step can be changed to any other text ranking models 

    Returns
    -------
    serp: `dict`
        Query and the key, and the retrieved result is the value.
        Result is a list of two lists: docid list and their scores.
    """
    # bm25 search
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # serp
    serp = {}
    for q in tqdm(queries):
        hits = searcher.search(q, k=args.k)
        results = [(hit.docid, hit.score) for hit in hits]
        serp[q] = list(map(list, zip(*results)))

    return serp
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--clariq", type=str)
    parser.add_argument("--collections", type=str)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    # search args
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--k1",type=float)
    parser.add_argument("--b", type=float)
    args = parser.parse_args()

    # clairq
    clariq_df = pd.read_csv(args.clariq, delimiter='\t')
    clariq_df.dropna(inplace=True) # 610 instances have no cq and ca
    queries = clariq_df['initial_request'].unique().tolist() + \
            clariq_df['question'].unique().tolist()

    # search
    clariq_serp = sparse_retrieve(queries, args)

    fout = open(args.output, 'w') 
    # add serp to clariq dataset
    for i,clariq_dict in clariq_df.to_dict('index').items():
        fout.write(json.dumps({
            "question": clariq_dict['initial_request'],
            "c_need": clariq_dict['clarification_need'],
            "c_question": clariq_dict['question'],
            "c_answer": clariq_dict['answer'],
            "q_serp": clariq_serp[clariq_dict['initial_request']],
            "cq_serp": clariq_serp[clariq_dict['question']],
        }, ensure_ascii=False)+'\n')
            # "facet": clariq_dict['facet_desc'],
            # "f_serp": clariq_serp[clariq_dict['facet_desc']],

    fout.close()
