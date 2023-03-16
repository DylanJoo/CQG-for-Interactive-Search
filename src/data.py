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

def clariq_cqg(path):
    """ Preprocess/synthesize CQG training (1st stage).

    Methods 
    -------
    get_top_n(): Retrieve the top n provenances of each example.
    """
    def get_top_n(ex, n):
        ex['provenance'] = ex['provenance'][:n]

    from datasets import load_dataset
    dataset = load_dataset('json', data_files=path)['train']

    if self.n_context is not None:
        dataset.map(get_top_n, fn_kwargs={"n": self.n_context})
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
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    is_train: Union[bool, str] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []

        ## Set batch input: B x L x H
        ## Pool retrieved 'title and passage': (B x N) L x H
        for batch in features:
            q = ex['question']
            for t, ctx in zip(ex['titles'], ex['provenances']):
                texts.append(f"question: {q} title: {t} context: {ctx}")

        inputs = self.tokenizer.batch_encode_plus(
                text, 
                max_length=self.max_length,
                pad_to_max_length=True,
                return_tensors=self.return_tensors,
                truncation=True
        )
        return inputs

