import os
from tqdm import tqdm
import json
import argparse

def lucene_search(searcher, queries, topk, ):

    hits = searcher.batch_search(
            queries, 
            qids, 
            k=args.k, 
            threads=args.num_threads)

