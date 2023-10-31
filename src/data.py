import torch
import random
import json
import numpy as np
import pandas as pd
from copy import copy

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PreTrainedTokenizerBase, 
        PaddingStrategy
)
from tools import TfidfTextScorer

@dataclass
class DataCollatorBase:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    max_src_length: Optional[int] = 512
    max_tgt_length: Optional[int] = 64

@dataclass
class DataCollatorForCQG(DataCollatorBase):
    is_train: Union[bool, str] = False
    n_contexts: Union[int, str] = 1
    scorer: Union[TfidfTextScorer] = None

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
                max_length=self.max_src_length,
                padding=True,
                return_tensors=self.return_tensors,
                truncation=True
        )

        ### adjustments
        inputs['input_ids'] = inputs['input_ids'].view(
                -1, self.n_contexts, inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
                -1, self.n_contexts * inputs['attention_mask'].size(-1)
        )

        ## labeling if needed.
        if self.is_train:
            targets = self.tokenizer.batch_encode_plus(
                    [ex['target'] for ex in features],
                    max_length=self.max_tgt_length,
                    padding=True,
                    return_tensors=self.return_tensors,
                    truncation=True,
            )
            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask

            if self.scorer is not None:
                inputs['label_weights'] = torch.zeros(
                        targets['input_ids'].shape
                )

                # 1. fetch tfidf scores
                scores = self.scorer.get_scores(
                        [ex['target'] for ex in features]
                )
                # 2. arrange the scores for the (tokenized) words
                for i in range(len(targets)):
                    word_idx_of_token = [idx if idx is not None else -1 \
                                    for idx in targets.word_ids(i)]
                    word_idx_of_token[word_idx_of_token.index(-1)] = -2
                    assert max(word_idx_of_token) == len(scores[i])-1, \
                            'Inconsistent length of scores and tokens'
                    inputs['label_weights'][i, :] = \
                            torch.tensor(scores[i]+[1, 0]).take(
                                    torch.tensor(word_idx_of_token)
                            )
        else:
            inputs['question'] =  [ex['question'] for ex in features]
            inputs['c_question'] =  [ex['target'] for ex in features]

        return inputs

@dataclass
class DataCollatorForMRG(DataCollatorBase):
    is_train: Union[bool, str] = False
    n_contexts: Union[int, str] = 1
    random_sample: Union[bool] = False
    enumerated_sample: Union[bool] = False
    answer_sample: Union[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        ## Set batch input: B x L x H
        ## Pool retrieved 'title and passage': (B x N) L x H
        texts = []
        for ex in features:
            q = ex['question']
            for t, ctx in zip(
                    ex['titles'][:self.n_contexts], 
                    ex['provenances'][:self.n_contexts]
            ):
                texts.append(f"question: {q} title: {t} context: {ctx}")

        if self.enumerated_sample:
            texts = texts * 2

        inputs = self.tokenizer.batch_encode_plus(
                texts, 
                max_length=self.max_src_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=self.truncation
        )

        ### adjustments
        inputs['input_ids'] = inputs['input_ids'].view(
                -1, self.n_contexts, inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
                -1, self.n_contexts * inputs['attention_mask'].size(-1)
        )

        texts = []
        ## labeling if needed.
        if self.is_train:
            # random sampling either of them
            if self.random_sample:
                acts = random.choices(['clarify','answer'], k=len(features))
                texts += [f"{act}: {ex['target'][act]}" \
                        for act, ex in zip(acts, features)]

            # enumerated sampling both of them; [c1, c2, ..] * 2
            if self.enumerated_sample:
                texts += [f"clarify: {ex['target']['clarify']}" \
                        for ex in features]
                texts += [f"answer: {ex['target']['answer']}" \
                        for ex in features]

            # answer sampling with only convQA setting
            if self.answer_sample:
                texts += [f"answer: {ex['target']['answer']}" \
                        for ex in features]

            targets = self.tokenizer.batch_encode_plus(
                    texts,
                    max_length=self.max_tgt_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=self.truncation,
            )
            target_ids = targets['input_ids']
            target_mask = targets['attention_mask'].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            inputs['labels'] = target_ids
            inputs['decoder_attention_mask'] = target_mask

        else:
            inputs['target'] =  [ex['target'] for ex in features]
            inputs['question'] =  [ex['question'] for ex in features]

        return inputs


