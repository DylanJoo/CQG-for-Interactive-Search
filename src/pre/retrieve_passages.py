import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset

def pack_clariq_to_jsonl(args):
    df = pd.read_csv(args.clariq, delimiter='\t')
    # 610 instances have no cq and ca
    clariq = Dataset.from_pandas(df.dropna())

    # add a column (question plus c_question)
    clariq = clariq.map(lambda ex: 
            {"q_and_cq": f"{ex['initial_request']} {ex['question']}"}
    )

    # Get queries for search
    queries = clariq['initial_request'] + clariq['q_and_cq']
    clariq_serp = retrieve(queries, args)

    # Add SERP to clariq dataset
    fout = open(args.output, 'w') 
    for clariq_dict in clariq:
        fout.write(json.dumps({
            "question": clariq_dict['initial_request'],
            "c_need": clariq_dict['clarification_need'],
            "c_question": clariq_dict['question'],
            "c_answer": clariq_dict['answer'],
            "q_serp": clariq_serp[clariq_dict['initial_request']],
            "ref_serp": clariq_serp[f"{clariq_dict['initial_request']} {clariq_dict['question']}"],
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
    canard_serp = retrieve(queries, args)
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
            "ref_serp": None, 
        }, ensure_ascii=False)+'\n')
    fout.close()

def pack_qrecc_to_jsonl(args):
    qrecc = load_dataset('json', data_files=args.qrecc)['train']
    ## [NOTE] Take out the NQ data since it's an augmented convQA

    # add column (unique question id)
    qrecc = qrecc.map(lambda ex: 
            {"id": f"{ex['Conversation_no']}_{ex['Turn_no']}"}
    )
    # add column (question plus answer)
    qrecc = qrecc.map(lambda ex: 
            {"q_and_a": f"{ex['Rewrite']}_{ex['Answer']}"}
    )

    # Get queries and search
    ## Setting 1: Rewritten queries
    queries = qrecc['Rewrite'] + qrecc['q_and_a']
    qrecc_serp = retrieve(queries, args)

    ## [NOTE] Other possible strategy:
    ## Conversational queries
    ## dense retrieve/cluster retrieve

    # Add SERP to qrecc dataset
    fout = open(args.output, 'w') 
    for qrecc_dict in qrecc:
        fout.write(json.dumps({
            "id": qrecc_dict['id'],
            "question": qrecc_dict['Rewrite'],
            "answer": qrecc_dict['Answer'],
            "q_serp": qrecc_serp[qrecc_dict['Rewrite']],
            "ref_serp": qrecc_serp[qrecc_dict['q_and_a']],
        }, ensure_ascii=False)+'\n')
    fout.close()

def retrieve(queries, args):

    if args.dense_retrieval:
        # FAISS search with DPR
        from pyserini.search import FaissSearcher, DprQueryEncoder
        query_encoder = DprQueryEncoder(args.q_encoder, device=args.device)
        searcher = FaissSearcher(args.index_dir, args.q_encoder)

        ## batching function
        def batch(iterable, n):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        batch_queries = batch(queries, args.batch_size)

        # serp
        serp = {}
        for qs in tqdm(batch_queries):
            ### use the query text as index.
            results = searcher.batch_search(qs, qs, k=args.k, threads=args.threads)
            results = [(id_, results[id_]) for id_ in qs]
            for topic, hits in results:
                serp[topic] = [hit for hit in hits if hit.docid != topic]
    else:
        # bm25 search
        from pyserini.search.lucene import LuceneSearcher
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
    parser.add_argument("--output", default='sample.jsonl', type=str)
    # search args
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--dense_retrieval", default=False, action='store_true')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--q-encoder", type=str, default='facebook/dpr-question_encoder-multiset-base')
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
        
