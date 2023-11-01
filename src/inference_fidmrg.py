import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from models import FiDT5
from tools import batch_iterator
from data_augmentation.loader import load_collections
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
            # [NOTE] No `ref_serp` when testing
            serp = data.pop('q_serp')[0][:args.top_k] 
            _ = data.pop('ref_serp', None)

            data.update({
                "titles": [titles[docid] for docid in serp],
                "provenances": [collections[docid] for docid in serp],
            })
            data_list.append(data)

    # load dataset
    dataset = Dataset.from_list(data_list)
    print(dataset)

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

            outputs = model.generate(**inputs, 
                                     max_length=args.max_length,
                                     num_beams=args.num_beams,
                                     do_sample=args.do_sample,
                                     top_k=args.top_k,
                                     return_dict_in_generate=True,
                                     output_scores=True)

            # output texts
            predictions = tokenizer.batch_decode(
                    outputs.sequences.detach().cpu(), 
                    skip_special_tokens=True
            )

            # output scores
            prediction_logits = torch.stack(outputs.scores, dim=1).cpu()
            prediction_logits = prediction_logits[:, 0, [1525, 17741]]
            answer_prob = prediction_logits.softmax(-1)[:, 0].tolist()
            clarify_prob = prediction_logits.softmax(-1)[:, 1].tolist()
            # >>> tokenizer.decode(17741)
            # 'clarify'
            # >>> tokenizer.decode(1525)
            # 'answer'
            # torch.stack(o.scores, dim=1)[:, 0, [1525, 17741]].softmax(-1)[:, 0]

            for i, prediction in enumerate(predictions):
                c_question, answer = "", ""
                if 'clarify' in prediction:
                    c_question = prediction.replace('clarify:', '').strip()
                elif 'answer' in prediction:
                    answer = prediction.replace('answer:', '').strip()
                else:
                    print('This instance has no `clarify` or `answer`')
                    print('-->', response)
                    c_question = "Neither"
                    answer = "Neither"

                f.write(json.dumps({
                    "qid": qids[i],
                    "question": batch_dataset['question'][i],
                    "answer": batch_dataset['answer'][i],
                    "response": {
                        "c_question": c_question, 
                        "answer": answer,
                        "prob_c_question": clarify_prob[i], 
                        "prob_answer": answer_prob[i],
                    },
                }, ensure_ascii=False)+'\n')

