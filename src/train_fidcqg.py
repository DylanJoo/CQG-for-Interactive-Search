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
from data import DataCollatorForCQG
from datasets import load_dataset
from trainers import Trainer
from arguments import *

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

    ## weighted generation loss from tfidf weight
    if model_args.tfidf_weighted:
        from utils import TfidfTextScorer
        tfidfscorer = TfidfTextScorer()
        tfidfscorer.fit_corpus(dataset['train']['targets'])

    ## data collator
    datacollator = DataCollatorForCQG(
            tokenizer=tokenizer, 
            padding=True,
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length,
            return_tensors='pt',
            n_contexts=model_args.n_contexts,
            is_train=True,
            scorer=tfidfscorer if model_args.tfidf_weighted else None
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
