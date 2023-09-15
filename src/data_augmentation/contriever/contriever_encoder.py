# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional

import torch
from transformers import AutoTokenizer
from pyserini.encode import DocumentEncoder
from pyserini.search.faiss import QueryEncoder
import sys
from .meta_contriever import Contriever

class ContrieverDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.model = Contriever.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or tokenizer_name)

    def encode(self, texts=None, titles=None, contents=None, max_length=256, **kwargs):
        if texts is None:
            if contents is None:
                texts = [title.strip() for title in titles]
            else:
                texts = [f'{title} {content}'.strip() for title, content in zip(titles, contents)]

        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        inputs.to(self.device)
        return self.model(**inputs).detach().cpu().numpy()


class ContrieverQueryEncoder(QueryEncoder):

    def __init__(self, model_name: str, tokenizer_name: str = None, device: str = 'cpu'):
        self.device = device
        self.model = Contriever.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or tokenizer_name)

    def encode(self, query: str, max_length: int = 64, **kwargs):
        inputs = self.tokenizer(
            [query],
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        inputs.to(self.device)
        embeddings = self.model(**inputs).detach().cpu().numpy()
        return embeddings.flatten()
