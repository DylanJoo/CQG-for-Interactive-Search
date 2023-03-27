import random
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)
from transformers import T5ForConditionalGeneration
from model import FiDT5
from data import clariq_cqg
from data import DataCollatorForCQG
from datasets import load_dataset

import os
os.environ["WANDB_DISABLED"] = "true"

@dataclass
class OurHFModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='t5-base')
    config_name: Optional[str] = field(default='t5-base')
    tokenizer_name: Optional[str] = field(default='t5-base')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class OurModelArguments:
    use_checkpoint: bool = field(default=False, metadata={
        "help": "use checkpoint in the encoder."
    })
    n_contexts: Optional[int] = field(default=3, metadata={
        "help": "the considered context (title and passage)", 
    })

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=0)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default='data/train_cqg_v0.sample.train.jsonl')
    eval_file: Optional[str] = field(default='data/train_cqg_v0.sample.eval.jsonl')
    max_length: int = field(default=256)

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./fidcqg')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=10000)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=False)
    # Customized arguments
    remove_unused_columns: bool = field(default=False)

def main():

    ## Parseing argument for huggingface packages
    parser = HfArgumentParser((OurHFModelArguments, OurModelArguments, OurDataArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        hfmodel_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ## additional config for models
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)
    t5 = T5ForConditionalGeneration.from_pretrained(hfmodel_args.model_name_or_path)
    model = FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    
    ## [REMOVE] add generation config
    # model.generation_config = generation_config
    model.set_checkpoint(model_args.use_checkpoint)

    ## dataset 
    ### [TODO] simplify this stage to load as N-passages-selection is moved to another stage.
    dataset = clariq_cqg(data_args.train_file, model_args.n_contexts)
    from datasets import disable_caching
    disable_caching()
    N = len(dataset['train'])
    if training_args.do_eval and data_args.eval_file is None:
        dataset['eval'] = dataset['train'].select(
                random.sample(range(N), 100)
        )
    else:
        dataset['eval'] = load_dataset('json', data_files=data_args.eval_file)['train']

    ## data collator
    datacollator = DataCollatorForCQG(
            tokenizer=tokenizer, 
            padding=True,
            max_length=data_args.max_length,
            return_tensors='pt',
            is_train=True,
            n_contexts=model_args.n_contexts
    )

    ## Trainer
    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'] if training_args.do_eval else None,
            data_collator=datacollator
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
