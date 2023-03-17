from transformers import AutoTokenizer
from src.data import clariq_cqg
from src.data import DataCollatorForCQG
from torch.utils.data import DataLoader


dataset = clariq_cqg('data/train_cqg_v0.sample.jsonl', 5)
print(dataset)
tokenizer = AutoTokenizer.from_pretrained('t5-base')

datacollator = DataCollatorForCQG(
        tokenizer=tokenizer, 
        padding=True,
        max_length=300,
        return_tensors='pt',
        is_train=True,
        n_contexts=5
)

dataloader = DataLoader(
        dataset['train'],
        batch_size=4,
        collate_fn=datacollator,
)

for d in dataloader:
    print(d.input_ids.shape)
    # print(d.input_ids)
