import random
import sys
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    GenerationConfig
)
from models import FiDT5
from data import DataCollatorForMRG
from datasets import load_dataset
from trainers import Trainer
from arguments import *

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
    train_file: Optional[str] = field(default='data/train_fidmrg_v0.sample.jsonl')
    eval_file: Optional[str] = field(default=None)
    max_length: int = field(default=256)
    max_length_answer: int = field(default=64)

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./fidmrg')
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
    parser = HfArgumentParser((HFModelArgs, ModelArgs, DataArgs, TrainingArgs))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        hfmodel_args, model_args, data_args, training_args = \
                parser.parse_args_into_dataclasses()

    ## additional config for models
    config = AutoConfig.from_pretrained(hfmodel_args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(hfmodel_args.tokenizer_name)
    model = FiDT5.from_pretrained(hfmodel_args.model_name_or_path)

    ## dataset 
    dataset = load_dataset('json', data_files=data_args.train_file,
            keep_in_memory=True)
    N = len(dataset['train'])
    if training_args.do_eval:
        dataset['dev'] = dataset['train'].select(random.sample(range(N), 100))

    ## data collator
    datacollator = DataCollatorForMRG(
            tokenizer=tokenizer, 
            padding=True,
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length,
            return_tensors='pt',
            n_contexts=model_args.n_contexts,
            is_train=True,
            random_sample=training_args.random_sample,
            enumerated_sample=training_args.enumerated_sample,
    )

    ## Trainer
    trainer = Trainer(
            model=model, 
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['dev'] if training_args.do_eval else None,
            data_collator=datacollator
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
