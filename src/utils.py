import collections
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np

# PATH='/home/jhju/CQG-for-Interactive-Search/data/clariq/train.tsv'
# tokenizer = T5Tokenizer.from_pretrained('t5-base')
#
# df = pd.read_csv(PATH, delimiter='\t').dropna()
# clariq = Dataset.from_pandas(df)
#
# corpus = clariq['question']
#
# vectorizer = TfidfVectorizer() # stop_words='english'
# X = vectorizer.fit_transform(corpus)
#
# word2id = vectorizer.vocabulary_
#
# for i, text in enumerate(corpus[:-1]):
#     # x = X[i, :].tocoo()
#     x = vectorizer.transform([text]).tocoo()
#     mapping = {k: v for k, v in zip(x.col, x.data)}
#     for w in text.split():
#         try: 
#             scores[i].append(mapping[word2id[w]])
#         except: # zero if out of vocab, e.g. stopwords.
#             scores[i].append(0)
#
# X = vectorizer.transform(corpus[:10])
# print(X.shape)
# for i in range(X.shape[0]):
#     x = X[i, :].tocoo()
#     mapping = {k: v for k, v in zip(x.col, x.data)}
#     for w in corpus[i].split():
#         try: 
#             scores[i].append((w, mapping[word2id[w]]))
#         except: # zero if out of vocab, e.g. stopwords.
#             scores[i].append((w, 0))

class TfidfTextScorer:

    def __init__(self, stop_words=None, norm='l2', **kwargs):
        self.vectorizer = TfidfVectorizer(
                norm=norm,
                stop_words=stop_words, 
                **kwargs
        )
        self.scores = collections.defaultdict(list)

    def fit_corpus(self, corpus):
        self.vectorizer.fit(corpus)
        self.word2idx = self.vectorizer.vocabulary_
        print("Total vocabulary: {}".format(len(self.word2idx)))

    def get_scores(self, texts):
        
        texts = list(texts) if isinstance(texts, str) else texts
        X = self.vectorizer.transform(texts)

        # fecth the scores of each tfidf vectors (text)
        scores = collections.defaultdict(list)
        for i in range(len(texts)):
            x = X[i, :].tocoo()
            text = texts[i]
            mapping = {k: v for k, v in zip(x.col, x.data)}
            for w in text.split():
                try:
                    scores[i].append(mapping[self.word2idx[w]])
                except:
                    scores[i].append(0)
        return scores
