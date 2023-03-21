import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from src.model import FiDT5

tokenizer = AutoTokenizer.from_pretrained('t5-small')
dataset = load_dataset(
        'json', data_files='data/train_cqg_v0.sample.jsonl'
)

model = FiDT5.from_pretrained('fidcqg/checkpoint-10000/')
model = model.to('cuda')
model.overwrite_forward_crossattention()
model.reset_score_storage()

from src.data import DataCollatorForCQG
datacollator = DataCollatorForCQG(
        tokenizer=tokenizer,
        padding=True,
        is_train=False,
        max_length=100,
        n_contexts=5
)

from torch.utils.data import DataLoader
dataloader = DataLoader(
        dataset['train'],
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        collate_fn=datacollator
)

# fout = open("removeme.txt", 'w')
# prediction 
with torch.no_grad():
    for batch in tqdm(dataloader):
        for k in batch:
            batch[k] = batch[k].cuda(model.device)

        model.reset_score_storage()
        outputs = model.generate(
                **batch,
                max_length=50
        )

        # reset 
        crossattention_scores = model.get_crossattention_scores(
                batch['attention_mask']
        )
        print(crossattention_scores)
        model.reset_score_storage()

        # write outputs
        for k, o in enumerate(outputs):
            c_question = tokenizer.decode(o, skip_special_tokens=True)
            # example = dataset.data[idx]
            print(c_question)
            # fout.write(c_question+'\n')

            """
            Evaluation codes
            """

