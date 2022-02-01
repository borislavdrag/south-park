import logging
from time import time
from datetime  import date

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pipelines import get_pipe
logger = logging.getLogger(' GrantClassifier ')
logging.basicConfig(level=logging.INFO)

from pipelines import *
from sklearn.feature_extraction.text import TfidfVectorizer


class GrantClassifier:

    def __init__(self, model_type, vec_method):
        self.model = get_pipe(model_type=model_type, vec_method=vec_method)
        self.encoder = None
        self.target = "Politikbereich"
        self.covariate = "Zweck"
        self.target_enc = None

        self.training_date = date.today().strftime("%d.%m.%Y")
        self.score = None  # TODO: determine eval metric

    def encode_target(self, df: pd.Series) -> pd.Series:
        self.encoder = LabelEncoder()
        self.encoder.fit(df)

        # self.target_enc = self.target + '_enc'
        return self.encoder.transform(df)

        # return df

    def evaluate(self, test_X: pd.Series, test_y: pd.Series):
        logger.info("To be implemented!!!")

    def train_and_test(self, df: pd.DataFrame, test_size: float = 0.1):
        logger.info("Training started")
        # logger.error(df.head())  # DEBUG
        train_X, test_X, train_y, test_y = train_test_split(df, 
                                                            df[self.target], 
                                                            test_size=test_size)  # TODO: stratified?
        train_y = self.encode_target(train_y)
        start = time()
        self.model.fit(train_X, train_y)
        logger.info("Trained in {:.2f}s".format(time() - start))
        self.evaluate(test_X, test_y)

    def predict(self, X: pd.Series):
        X[self.target + '_pred'] = self.encoder.inverse_transform(self.model.predict(X))


if __name__ == '__main__':
    show_model = GrantClassifier(model_type='xgboost', vec_method='tfidf')
    data = pd.read_excel("data/data2020.xlsx").head(100)
    show_model.train_and_test(data)

