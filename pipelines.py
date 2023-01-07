import xgboost as xgb
from sklearn import naive_bayes as nb

from transformers import KeepColumns, CleanData
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer 


def get_pipe(model_type='xgboost', vec_method='tfidf'):

    vector_pipe = {
        'tfidf': TfidfVectorizer(max_features=500),

      #  'word2vec':  # DLC
    }

    model_pipe = {
        'xgboost': Pipeline([
        ('columns', KeepColumns(cols=['line'])),
        ('preprocess', CleanData(covariate='line')), 
        ('vectorizer', vector_pipe[vec_method]),
        ('clf', xgb.XGBClassifier(objective='multi:softprob',
                                  eval_metric='mae',
                                  max_depth=7,
                                  n_estimators=200,
                                  learning_rate=0.1,
                                  use_label_encoder=False,
                                  n_jobs=8,
                                  verbosity=1))
        ]),
        'bayes': Pipeline([
        ('columns', KeepColumns(cols=['line'])),
        ('preprocess', CleanData(covariate='line')), 
        ('vectorizer', vector_pipe[vec_method]),
        ('clf', nb.MultinomialNB())
        ]),

        #  'cnn':  # DLC
    }

    return model_pipe[model_type]

# Note: in a productive application pipelines won't be used as much to
# select type of model, as to select a model with other parameters or
# data types: depending on input, maximizing variable, filter of data

