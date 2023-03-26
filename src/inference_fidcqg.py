import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from model import FiDT5
from data import DataCollatorForCQG
from torch.utils.data import DataLoader

def generate(inputs, max_length, device, model, tokenizer, f):

    source_questions = batch.pop('question')
    target = batch.pop('c_question')
    for k in inputs:
        inputs[k] = inputs[k].cuda(device)

    outputs = model.generate(**inputs, max_length=max_length)
    scores = model.get_crossattention_scores(batch['attention_mask']).tolist()

    # write outputs
    for k, o in enumerate(outputs):
        c_question_pred = tokenizer.decode(o, skip_special_tokens=True)
        f.write(json.dumps({
            "question": source_questions[k],
            "prediction": c_question_pred,
            "score_mu": np.mean(scores[k]),
            "score_sigma": np.std(scores[k]),
            "scores": scores[k],
        }, ensure_ascii=False)+'\n')

        # Evaluation codes if existed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--used_checkpoint", type=str, default=None)
    parser.add_argument("--used_tokenizer", type=str, default=None)
    parser.add_argument("--calculate_crossattention", action='store_true', default=False)
    parser.add_argument("--n_contexts", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    # load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.used_tokenizer)
    dataset = load_dataset('json', data_files=args.jsonl_file)['train']

    model = FiDT5.from_pretrained(args.used_checkpoint).to(args.device)
    model.overwrite_forward_crossattention()
    model.reset_score_storage()

    # organize dataset/dataloader
    datacollator = DataCollatorForCQG(
            tokenizer=tokenizer,
            padding=True,
            is_train=False,
            max_length=args.max_length,
            n_contexts=args.n_contexts
    )
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=False,
            collate_fn=datacollator
    )

    # predict and write
    with torch.no_grad(), open(args.output_file, 'w') as f:
        for batch in tqdm(dataloader):

            model.reset_score_storage()
            generate(inputs=batch, 
                     max_length=args.max_length,
                     device=args.device,
                     model=model, 
                     tokenizer=tokenizer,
                     f=f)
            model.reset_score_storage()

