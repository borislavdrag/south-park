import logging
import joblib
from time import time
from datetime import date

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pipelines import get_pipe
logger = logging.getLogger('GrantClassifier')
logging.basicConfig(level=logging.INFO)


class GrantClassifier:

    def __init__(self, model_type, vec_method):
        self.model = get_pipe(model_type=model_type, vec_method=vec_method)
        self.encoder = None
        self.target = "Politikbereich"
        self.covariate = "Zweck"

        self.training_date = date.today().strftime("%d.%m.%Y")
        self.score = None  # TODO: determine eval metric

    def encode_target(self, df: pd.Series) -> pd.Series:
        self.encoder = LabelEncoder()
        self.encoder.fit(df)

        return self.encoder.transform(df)

    def evaluate(self, test_X: pd.DataFrame, test_y: pd.Series):
        test_y = self.encoder.transform(test_y)
        predictions = self.model.predict(test_X)

        # TODO: f1 and auc need fixing
        accuracy = accuracy_score(predictions, test_y)
        f1_metric = f1_score(predictions, test_y, average='micro')  # micro because of class imbalance
        # auc_metric = roc_auc_score(test_y, self.model.predict_proba(test_X), multi_class='ovr')
        auc_metric = '?'  # TODO
        logger.info("\nAccuracy: {}\nF1: {}\nAUC: {}".format(accuracy, 
                                                           f1_metric, 
                                                           auc_metric))
        self.score = accuracy

    def train_and_test(self, df: pd.DataFrame, test_size: float = 0.1):
        logger.info("Training started")
        train_X, test_X, train_y, test_y = train_test_split(df, 
                                                            df[self.target], 
                                                            test_size=test_size,
                                                            random_state=359,
                                                            stratify=df[[self.target]])
        train_y = self.encode_target(train_y)
        start = time()
        self.model.fit(train_X, train_y)
        logger.info("Trained in {:.2f}s".format(time() - start))
        self.evaluate(test_X, test_y)

    def predict(self, X: pd.Series):
        predictions = self.encoder.inverse_transform(self.model.predict(X))

        # manual rules:
        predictions = np.where(X[self.covariate].str.contains("Kosten für die Beschäftigung"), "Sport", predictions)

        return predictions


    def save_model(self, filename: str):
        if not filename:
            filename = "models/grantclf_latest.pkl"

        try:
            joblib.dump(self, filename)
            logger.info("Model saved to {}".format(filename))
        except Exception as e:
            logger.error("Saving failed with {}: {} ".format(e.__class__.__name__, e))


if __name__ == '__main__':

    # Test example for the model with very few data points:
    print("Test example fot Grant Classifier:")
    show_model = GrantClassifier(model_type='xgboost', vec_method='tfidf')
    data = pd.read_excel("data/data2020.xlsx")
    data_small = data.head(1000)

    values = data_small['Politikbereich'].value_counts()
    single_values = values[values == 1]
    # having just one sample prevents stratified samples and doesn't bring added value
    data_small = data_small[~data_small['Politikbereich'].isin(single_values.index)]

    show_model.train_and_test(data_small)
    print("Sample predictions:")
    print(show_model.predict(data.tail(10)))
    print(show_model.predict(data[data["Zweck"].str.contains("Kosten für die Be")].head(2)))

