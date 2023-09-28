from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class HFModelArgs:
    model_name_or_path: Optional[str] = field(default='t5-base')
    config_name: Optional[str] = field(default='t5-base')
    tokenizer_name: Optional[str] = field(default='t5-base')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class ModelArgs:
    use_checkpoint: bool = field(default=False)
    n_contexts: Optional[int] = field(default=3)
    tfidf_weighted: bool = field(default=False)
    tfidf_weighted_stopwords: Optional[str] = field(default=None)

@dataclass
class DataArgs:
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=0)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default='data/train_cqg_v0.sample.train.jsonl')
    eval_file: Optional[str] = field(default=None)
    max_src_length: int = field(default=256)
    max_tgt_length: int = field(default=64)

@dataclass
class TrainingArgs(TrainingArguments):
    output_dir: str = field(default='./fidcqg')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=10000)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='steps')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=False)
    # Customized arguments
    remove_unused_columns: bool = field(default=False)
