# Author: Dylan
# This source code is edited based on the FiD's code.
# The FiD's codes are made for standard QA tasks.
# And this code is made for Conversational QA.
# 
## Copyright (c) Facebook, Inc. and its affiliates.
## All rights reserved.
##
## This source code is licensed under the license found in the
## LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
import pandas as pd

def clariq_cqg(path, n_context):
    """ Preprocess/synthesize CQG training (1st stage).

    Methods 
    -------
    get_top_n(): Retrieve the top n provenances of each example.
    """
    def get_top_n(ex, n):
        ex['provenances'] = ex['provenances'][:n]

    from datasets import load_dataset
    dataset = load_dataset('json', data_files=path)

    if n_context is not None:
        dataset.map(get_top_n, fn_kwargs={"n": n_context})
    return dataset


from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PreTrainedTokenizerBase, 
        PaddingStrategy
)

@dataclass
class DataCollatorForCQG:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    max_length_answer: Optional[int] = 64
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    is_train: Union[bool, str] = False
    n_contexts: Union[int, str] = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        ## Set batch input: B x L x H
        ## Pool retrieved 'title and passage': (B x N) L x H
        texts = []
        for ex in features:
            q = ex['question']
            for t, ctx in zip(ex['titles'][:self.n_contexts], ex['provenances'][:self.n_contexts]):
                texts.append(f"question: {q} title: {t} context: {ctx}")

        inputs = self.tokenizer.batch_encode_plus(
                texts, 
                max_length=self.max_length,
                padding=True,
                return_tensors=self.return_tensors,
                truncation=True
        )

        ### adjustments
        inputs['input_ids'] = inputs['input_ids'].view(
                -1, self.n_contexts, inputs.input_ids.size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
                -1, self.n_contexts, inputs.attention_mask.size(-1)
        )

        ## labeling if needed.
        if self.is_train:
            targets = self.tokenizer.batch_encode_plus(
                    [ex['c_question'] for ex in features],
                    max_length=self.max_length_answer if self.max_length_answer > 0 else False,
                    padding=True,
                    return_tensors=self.return_tensors,
                    truncation=True,
            )
            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask
        ## this is for comparisons
        else:
            inputs['c_question'] =  [ex['c_question'] for ex in features]
            inputs['question'] =  [ex['question'] for ex in features]

        return inputs

