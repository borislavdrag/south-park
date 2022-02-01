import spacy
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class KeepColumns(BaseEstimator, TransformerMixin):
    """
    Keeps only useful columns of a pandas DataFrame.
    """
    def __init__(self, cols):
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame):
        Y = X.copy()
        return Y[self.cols]

class CleanData(BaseEstimator, TransformerMixin):
    """
    Preprocesses text data needed for classification:
    * removes punctuation, digits, umlauts
    * lowercases text
    * lemmatizes words
    * removes stop words

    Returns the data in sentence format.
    """
    def __init__(self, covariate: str):
        self.covariate = covariate
        
        self.umlauts = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
        self.nlp = spacy.load('de_core_news_sm')
        self.stopwords = self.nlp.Defaults.stop_words

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame):
        X[self.covariate] = X[self.covariate] \
                             .str.replace('[^\w\s]+', ' ', regex=True) \
                             .str.replace('\d+', ' ', regex=True) \
                             .str.replace(' +', ' ', regex=True) \
                             .replace(self.umlauts, regex=True) \
                             .str.lower() \
                             .apply(lambda x: [word.lemma_ for word in self.nlp(x)]) \
                             .apply(lambda x: [item for item in x if item not in self.stopwords]) \
                             .str.join(' ')

        return X[self.covariate]

