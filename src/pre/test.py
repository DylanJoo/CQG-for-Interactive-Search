import collections
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np

from transformers import T5Tokenizer

PATH='/home/jhju/CQG-for-Interactive-Search/data/clariq/train.tsv'
tokenizer = T5Tokenizer.from_pretrained('t5-base')

df = pd.read_csv(PATH, delimiter='\t').dropna()
clariq = Dataset.from_pandas(df)

corpus = clariq['question']

vectorizer = TfidfVectorizer() # stop_words='english'
X = vectorizer.fit_transform(corpus)

scores = collections.defaultdict(list)
word2id = vectorizer.vocabulary_

for i, text in enumerate(corpus[:-1]):
    x = X[i, :].tocoo()
    mapping = {k: v for k, v in zip(x.col, x.data)}
    for w in tokenizer.tokenize(text):
        try: 
            scores[i].append((w, mapping[word2id[w]]))
        except: # zero if out of vocab, e.g. stopwords.
            scores[i].append((w, 0))



