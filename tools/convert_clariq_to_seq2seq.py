import random
import json
import argparse
import collections
import os

def make_cqg_dataset(args):
    
    with open(args.path_multiturn_jsonl, 'r') as fin, open(args.path_output, 'w') as fout:
        for i, line in enumerate(fin):
            multiturn = json.loads(line.strip())
            init_question = multiturn['init_request']
            question = multiturn['question']
            # implicit needs of (historical questions and candidate passages)
            implicit_keywords = multiturn['implicit_keywords']
            random.shuffle(implicit_keywords)
            keywords_implicit = ", ".join(implicit_keywords)

            # answer = multiturn['answer'] # in this task, we would not use the answer replied.
            context = []
            context.append(multiturn['user_utterances'].pop(0))
            for u, s in zip(multiturn['user_utterances'], multiturn['system_responses']):
                context.append(s)
                context.append(u)

            context = "|||".join(context)

            if 'implicit' in args.path_output:
                fout.write(
                        f"Context: {context} Keywords: {keywords_implicit} Clarifying:\t{question}\n"
                )
            else:
                # simple generation 
                fout.write(
                        f"Context: {context} Clarifying:\t{question}\n"
                )

        if i % 1000 == 0:
            print("{} finished...".format(i))


parser = argparse.ArgumentParser()
parser.add_argument("-clariq", "--path_multiturn_jsonl", default="data/clariq.multiturn.train.synthetic.jsonl", type=str)
parser.add_argument("-t5_output", "--path_output", default="data/clariq.train.cqg.t5.tsv", type=str)
parser.add_argument("--keyword_based", default=False, action='store_true')
args = parser.parse_args()

make_cqg_dataset(args)


