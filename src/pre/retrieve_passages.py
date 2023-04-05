import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher

def pack_clariq_to_jsonl(args):
    clariq_df = pd.read_csv(args.clariq, delimiter='\t')
    clariq_df.dropna(inplace=True) # 610 instances have no cq and ca

    # Get queries
    queries = clariq_df['initial_request'].unique().tolist() + \
            clariq_df['question'].unique().tolist()

    # Search
    clariq_serp = sparse_retrieve(queries, args)

    # Add SERP to clariq dataset
    fout = open(args.output, 'w') 
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

def pack_canard_to_jsonl(args):
    canard = load_dataset('json', data_files=args.canard)['train']

    # Get queries
    ## Setting 1: Rewritten queries
    queries = canard['rewrite']
    ## [NOTE] Setting 2: Conversational queries
    # queries = ??

    # Search 
    ## Setting 1: TopK provenances
    canard_serp = sparse_retrieve(queries, args)
    ## [NOTE] Setting 2: dense retrieve/cluster retrieve
    # canard_serp = ??

    # Add SERP to canard dataset
    fout = open(args.output, 'w') 
    for canard_dict in canard:
        fout.write(json.dumps({
            "id": canard_dict['id'],
            "question": canard_dict['rewrite'],
            "answer": canard_dict['answer'],
            "q_serp": canard_serp[canard_dict['rewrite']],
        }, ensure_ascii=False)+'\n')
    fout.close()

def pack_qrecc_to_jsonl(args):
    qrecc = load_dataset('json', data_files=args.qrecc)['train']
    ## [NOTE] Take out the NQ data since it's an augmented convQA
    qrecc = qrecc.filter(lambda ex: ex['Conversation_source'] != 'nq')

    # add column (unique question id)
    qrecc = qrecc.map(lambda ex: 
            {"id": f"{ex['Conversation_no']}_ex['Turn_no']"}
    )

    # Get queries
    ## Setting 1: Rewritten queries
    queries = qrecc['Rewrite']
    ## [NOTE] Setting 2: Conversational queries
    # queries = ??

    # Search 
    ## Setting 1: TopK provenances
    qrecc_serp = sparse_retrieve(queries, args)
    ## [NOTE] Setting 2: dense retrieve/cluster retrieve
    # qrecc_serp = ??

    # Add SERP to qrecc dataset
    fout = open(args.output, 'w') 
    for qrecc_dict in qrecc:
        fout.write(json.dumps({
            "id": qrecc_dict['id'],
            "question": qrecc_dict['Rewrite'],
            "answer": qrecc_dict['Answer'],
            "q_serp": qrecc_serp[qrecc_dict['Rewrite']],
        }, ensure_ascii=False)+'\n')
    fout.close()

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
    parser.add_argument("--clariq", type=str, default=None)
    parser.add_argument("--canard", type=str, default=None)
    parser.add_argument("--qrecc", type=str, default=None)
    parser.add_argument("--collections", type=str, default=None)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    # search args
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--k1",default=0.9, type=float)
    parser.add_argument("--b", default=0.4, type=float)
    args = parser.parse_args()

    if args.clariq:
        pack_clariq_to_jsonl(args)

    if args.canard:
        pack_canard_to_jsonl(args)

    if args.qrecc:
        pack_qrecc_to_jsonl(args)
        

