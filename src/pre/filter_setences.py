from sklearn.feature_extraction.text import TfidfVectorizer  
import numpy as np
import pandas as pd 


class SentFilter:
    """ 
    Filter words in the sentence that subjectively unimportant among texts.
    With unsupervised TFIDF, we fit the given copurs (a set of sentences);
    hopefully, the filter can drop the unwanted(uninformative) tokens.
    """
    def __init__(self):
        self.mapping

    def initailize_with_tfidf(self, corpus, **tfidf_kwargs):
        """
        Set the pre-fitted tfidf vector
        """
        # stop_words = 'english' 
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self.vectorizer.fit(['unk'] + corpus)
        self.idf = vectorizer.idf_
        self.word2id = vectorizer.vocabulary_
        self.id2word = {v: k for k, v in vectorizer.vocabulary_}

    def get_scores(self, X):
        scores = collections.defaultdict(list)
        for w in text.split():
            scores[i].append(X[i, self.word2id.get(w, 'unk')])

        self.vectorizer.transform(X)
    # vectorizer.transform(["interesting is interested"]).toarray()[0][vectorizer.vocabulary_['interesting']]
    def filter(self, texts):
        self.vectorizer.transform(texts)
        return 0
