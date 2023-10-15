import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from models import FiDT5
from data import DataCollatorForCQG
from tools import batch_iterator
from data_augmentation.loader import load_collections
from torch.utils.data import DataLoader
from datasets import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--collections", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--used_checkpoint", type=str, default=None)
    parser.add_argument("--used_tokenizer", type=str, default=None)
    parser.add_argument("--calculate_crossattention", action='store_true', default=False)
    parser.add_argument("--n_contexts", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--device", type=str, default='cuda')
    # generation configs
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action='store_true', default=False)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # build datset
    collections, titles = load_collections(args.collections, title=True)
    data_list = []
    with open(args.jsonl_file, 'r') as fin:
        for line in tqdm(fin):
            data = json.loads(line.strip())
            # [NOTE] the diverse set of serp should be considered
            serp = data.pop('q_serp')[0][:args.top_k] 
            _ = data.pop('ref_serp')

            data.update({
                "titles": [titles[docid] for docid in serp],
                "provenances": [collections[docid] for docid in serp],
            })
            data_list.append(data)

    # load dataset
    dataset = Dataset.from_list(data_list)

    # load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.used_tokenizer)
    model = FiDT5.from_pretrained(args.used_checkpoint).to(args.device)

    # predict and write
    with torch.no_grad(), open(args.output_file, 'w') as f:
        for batch_dataset in tqdm(
                batch_iterator(dataset, args.batch_size),
                total=(len(dataset)//args.batch_size)+1
        ):

            qids = batch_dataset.pop('id')
            texts = []
            for i in range(len(qids)):
                q = batch_dataset['question'][i]
                # batch_titles = batch_dataset['titles'][:args.n_contexts]

                for t, ctx in zip(
                        batch_dataset['titles'][0][:args.n_contexts], 
                        batch_dataset['provenances'][0][:args.n_contexts]
                ):
                    texts.append(f"question: {q} title: {t} context: {ctx}")

            inputs = tokenizer.batch_encode_plus(
                    texts, 
                    max_length=256,
                    padding=True,
                    return_tensors='pt',
                    truncation=True
            ).to(args.device)

            inputs['input_ids'] = inputs['input_ids'].view(
                    -1, args.n_contexts, inputs['input_ids'].size(-1)
            )
            inputs['attention_mask'] = inputs['attention_mask'].view(
                    -1, args.n_contexts * inputs['attention_mask'].size(-1)
            )

            outputs = model.generate(
                    **inputs, 
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    top_k=args.top_k
            ).detach().cpu()
            predictions = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
            )

            for i, prediction in enumerate(predictions):
                f.write(json.dumps({
                    "qid": qids[i],
                    "question": batch_dataset['question'][i],
                    "c_question": prediction
                }, ensure_ascii=False)+'\n')

