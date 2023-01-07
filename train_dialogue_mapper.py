import pandas as pd
import glob
from datetime import date
import logging
logger = logging.getLogger('DialogueMapper/Training')
logging.basicConfig(level=logging.INFO)

from DialogueMapper import DialogueMapper


if __name__ == '__main__':
    all_files = glob.glob("data/lines*.csv")
    datasets = list()

    for filename in all_files:
        df = pd.read_csv(filename, header=0,
                         names=["title", "character", "line"])
        datasets.append(df)

    logger.info("Preparing data for training...")
    data = pd.concat(datasets, axis=0, ignore_index=True)
    values = data['character'].value_counts()
    single_values = values[values == 1]
    # having just one sample prevents stratified samples and doesn't bring added value
    data = data[~data['character'].isin(single_values.index)]

    # logger.info("Training xgboost model...")
    # model_xgboost = DialogueMapper(model_type='xgboost', vec_method='tfidf')
    # model_xgboost.train_and_test(data)
    # model_xgboost.save_model("models/southpark_xgb_{}.pkl".format(date.today().strftime("%d-%m-%Y")))

    logger.info("Training NaiveBayes model...")
    model_bayes = DialogueMapper(model_type='bayes', vec_method='tfidf')
    model_bayes.train_and_test(data)
    model_bayes.save_model("models/southpark_nb_{}.pkl".format(date.today().strftime("%d-%m-%Y")))
    

