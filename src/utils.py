from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np

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
        return scoresass
