import re
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def aggregate_ff(qa, qcq, writer):
    """ Aggregate two source of data in 50/50."""

    writer.write(json.dumps({
        "question": qa['question'],
        "mi_response": qa['answer'],
        "c_need": 0, 
        "titles": qa['titles'],
        "provenances": qa['provenances'],
    }, ensure_ascii=False)+'\n')

    # clarification question
    if qcq is not None:
        assert qcq['question'] == qa['question'], \
                "the question is not aligned."
        f.write(json.dumps({
            "question": qa['question'],
            "mi_response": qcq['prediction'],
            "c_need": 1,
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
            # setting 0: direct target
            # aggregate  randomly without modification
            aggregate_ff(
                    qa=convqa[i],
                    qca=convqcq[i] if args.convqcq else None,
                    writer=fout
            )

            # [NOTE] Other possible settings: 
            # by c_need
            # by scores
            # by other relevance
            # by whether or not has the exact match words
