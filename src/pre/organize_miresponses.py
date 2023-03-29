import re
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def aggregate_ff(qa, qcq, f):
    """ 
    Aggregate two source of data in 50/50.  
    """
    f.write(json.dumps({
        "question": qa['question'],
        "mi_response": qa['answer'],
        "titles": qa['titles'],
        "provenances": qa['provenances']
    }, ensure_ascii=False)+'\n')

    # clarification question
    if qcq is not None:
        assert qcq['question'] == qa['question'], \
                "the question is not aligned."
        f.write(json.dumps({
            "question": qa['question'],
            "mi_response": qcq['prediction'],
            "titles": qa['titles'],
            "provenances": qa['provenances']
        }, ensure_ascii=False)+'\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--convqa", type=str, default=None)
    parser.add_argument("--convqcq", type=str, default=None)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    args = parser.parse_args()

    # Load ConvQA's (i.e., CANARD+QuAC)
    convqa = load_dataset('json', data_files=args.convqa)['train']

    if args.convqcq:
        convqcq = load_dataset('json', data_files=args.convqcq)['train']
        assert len(convqa) == len(convqcq), 'Inconsistent length of data'

    # Combine the two sources
    with open(args.output, 'w') as fout:
        for i in tqdm(range(len(convqa))):
            data_convqa = convqa[i]
            data_convqcq = convqcq[i] if args.convqcq else None

            # Setting 1: aggregate two of them with 50/50.
            aggregate_ff(data_convqa, data_convqcq, fout)

            # [NOTE] Other possible settings: 
            # by c_need
            # by scores
            # by other relevance
            # by whether or not has the exact match words
