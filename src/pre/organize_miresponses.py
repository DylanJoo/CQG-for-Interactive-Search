import re
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def aggregate_ff(data, cq, f):
    """ 
    Aggregate two source of data in 50/50.  
    """
    assert cq['question'] == data['question'], "the question is not aligned."
    print(data.keys())
    print(cq.keys())
    f.write(json.dumps({
        "question": data['question'],
        "mi_response": data['c_question'],
        "titles": data['titles'],
        "provenances": data['provenances']
    }, ensure_ascii=False)+'\n')

    # clarification question
    f.write(json.dumps({
        "question": data['question'],
        "mi_response": cq['prediction'],
        "titles": data['titles'],
        "provenances": data['provenances']
    }, ensure_ascii=False)+'\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--convqa", type=str)
    parser.add_argument("--convqcq", type=str)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    args = parser.parse_args()

    # Load ConvQA's (i.e., CANARD+QuAC)
    convqa = load_dataset('json', data_files=args.convqa)['train']
    convqcq = load_dataset('json', data_files=args.convqcq)['train']
    assert len(convqa) == len(convqcq), 'Inconsistent length of data'

    # Combine the two sources
    with open(args.output, 'w') as fout:
        for i in tqdm(range(len(convqa))):
            data_convqa = convqa[i]
            data_convqcq = convqcq[i]

            # Setting 1: aggregate two of them with 50/50.
            aggregate_ff(data_convqa, data_convqcq, fout)

            # [NOTE] Other possible settings: 
            # by c_need
            # by scores
            # by other relevance
            # by whether or not has the exact match words
