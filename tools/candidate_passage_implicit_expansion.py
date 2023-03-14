import json
import argparse
import math
import nltk
from collections import Counter
from pyserini.search.lucene import LuceneSearcher
from pyserini.index import IndexReader

parser = argparse.ArgumentParser()
parser.add_argument("-k_docs", "--topk_docs_retrieval", default=30, type=int)
parser.add_argument("-k1", "--k1", default=float(0.9), type=float)
parser.add_argument("-b", "--b", default=float(0.4), type=float)
parser.add_argument("-k_keywords", "--topk_terms", default=10, type=int)
parser.add_argument("-index", "--dir_index", default=None, type=str)
parser.add_argument("-query", "--path_query", default=None, type=str)
parser.add_argument("-output", "--path_output", default=None, type=str)
args = parser.parse_args()

# NLTK stopwords
STOPWORDS = nltk.corpus.stopwords.words('english')
# Lucene index reader
READER = IndexReader(args.dir_index)

def entropy_based_ranking(hits, excluded=[]):

    def calculate_entropy(df):
        p = df / args.topk_docs_retrieval
        if p == 1:
            return math.inf
        else:
            return p * math.log(p) + (1-p) * math.log(1-p)

    # collection implicit keywords from top30 docuemnts
    term_dfs = Counter()
    for i in range(len(hits)):
        term_dfs += Counter(READER.get_document_vector(hits[i].docid).keys())

    # remove explicit keywords (need)
    for kw in (excluded + STOPWORDS):
        del term_dfs[kw]

    # pos-tag filtering
    for (kw, pos) in nltk.pos_tag(term_dfs, tagset='universal'):
        if pos not in ['NOUN', 'ADJ']:
            del term_dfs[kw]

    # sort by entropy
    termlist_ranked = [
            term for (term, df) in sorted(term_dfs.items(), key=lambda x: calculate_entropy(x[1]))
    ]

    return termlist_ranked

def conversational_query_expansion(args):

    # Lucuene initialization
    searcher = LuceneSearcher(args.dir_index)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # Load query text and query ids
    data_dict = {}
    with open(args.path_query, 'r') as fin:
        for line in fin:
            data = json.loads(line.strip())
            data_dict[data.pop('record_id')] = data

    # prepare output
    with open(args.path_output, 'w') as fout:
        # search for each q
        for qi, (qidx, data) in enumerate(data_dict.items()):
            # initial request and explicit information need
            query = data['init_request'] + " ".join([w for w in data['explicit_keywords']])
            hits = searcher.search(query, k=args.topk_docs_retrieval)

            # entropy based ranking
            kw_implicit_candidates = entropy_based_ranking(hits, data['explicit_keywords'])

            # add implicit terms and truncate at 10
            kw_implicit = data['implicit_keywords'] + kw_implicit_candidates
            data.update({"implicit_keywords": kw_implicit[:args.topk_terms]})

            fout.write(json.dumps(data) + '\n')

            if qi % 10000 == 0:
                print(data['init_request'])
                print(" | ".join(data['explicit_keywords']))
                print(" | ".join(data['implicit_keywords']))
                print(data['question'])

conversational_query_expansion(args)
print("Done")
