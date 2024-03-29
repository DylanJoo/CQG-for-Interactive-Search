import json
import argparse
from tqdm import tqdm
import numpy as np
import loader

def dense_retrieve(queries, args):
    import torch
    from pyserini.search import FaissSearcher
    from contriever import ContrieverQueryEncoder
    query_encoder = \
            ContrieverQueryEncoder(args.q_encoder, device=args.device)
    searcher = FaissSearcher(args.index_dir, args.q_encoder)
    if torch.cuda.is_available():
        searcher.query_encoder.model.to(args.device)
        searcher.query_encoder.device = args.device

    # serp
    batch_queries = []
    serp = {}
    for index, q in enumerate(tqdm(queries, total=len(queries))):
        batch_queries.append(q)
        # form a batch
        if (len(batch_queries) % args.batch_size == 0 ) or \
                (index == len(queries) - 1):
            results = searcher.batch_search(
                    batch_queries, batch_queries, 
                    k=args.k, threads=args.threads
            )

            for q_ in tqdm(batch_queries):
                result = [(hit.docid, float(hit.score)) for hit in results[q_]]
                serp[q_] = list(map(list, zip(*result)))

            # clear
            batch_queries.clear()
            results.clear()
        else:
            continue
    return serp

def sparse_retrieve(queries, args):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # retrieval
    serp = {}
    for q in tqdm(queries):
        hits = searcher.search(q, k=args.k)
        results = [(hit.docid, hit.score) for hit in hits]
        serp[q] = list(map(list, zip(*results)))

    # pack datasets
    return serp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--clariq", type=str, default=None)
    parser.add_argument("--convqa", type=str, default=None)
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
        dataset = loader.clariq(args.clariq)

        # pool queries for search
        queries = np.unique(dataset['question']).tolist()
        print(f"Amount of queries (request): {len(queries)}")
        queries += np.unique(dataset['q_and_cq']).tolist()
        print(f"Amount of queries (request and request+clariq): {len(queries)}")

        # SERP for clariq dataset
        if args.dense_retrieval:
            serp = dense_retrieve(queries, args)
        else:
            serp = sparse_retrieve(queries, args)

        fout = open(args.output, 'w') 
        for clariq_dict in dataset:
            fout.write(json.dumps({
                "question": clariq_dict['question'],
                "c_question": clariq_dict['c_question'],
                "q_serp": serp[clariq_dict['question']],
                "ref_serp": serp[clariq_dict['q_and_cq']],
                "c_answer": clariq_dict['answer'],
                "c_need": clariq_dict['clarification_need'],
            }, ensure_ascii=False)+'\n')
        fout.close()

    if args.convqa:
        if 'qrecc' in args.convqa:
            dataset = loader.qrecc(args.convqa)
        if 'topiocqa' in args.convqa:
            # In Topiocqa, it contains the 'rationale' for gold passage searching.
            dataset = loader.topiocqa(args.convqa, reference_key='rationale')

        # pool queries for search
        queries = np.unique(dataset['question']).tolist()
        print(f"Amount of queries (request): {len(queries)}")
        queries += np.unique(dataset['q_and_a']).tolist()
        print(f"Amount of queries (request and request+answer): {len(queries)}")

        # SERP for qrecc dataset
        if args.dense_retrieval:
            serp = dense_retrieve(queries, args)
        else:
            serp = sparse_retrieve(queries, args)

        # Add SERP to qrecc dataset
        fout = open(args.output, 'w') 
        for convqa_dict in dataset:
            fout.write(json.dumps({
                "id": convqa_dict['id'],
                "question": convqa_dict['question'],
                "answer": convqa_dict['answer'],
                "q_serp": serp[convqa_dict['question']],
                "ref_serp": serp[convqa_dict['q_and_a']],
            }, ensure_ascii=False)+'\n')
        fout.close()
