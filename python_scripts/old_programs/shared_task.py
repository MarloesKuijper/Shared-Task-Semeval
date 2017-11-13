import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from pandas import DataFrame
import numpy as np
from scipy import sparse
import re
from scipy.stats.stats import pearsonr   
import pandas as pd
import gensim, logging
from numpy import mean, std
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

df_anger_en_train = pd.read_csv("data/en/EI-reg-en_anger_train.txt", header=None, names=["text", "emotion", "intensity"], sep="\t")
df_anger_en_dev = pd.read_csv("data/en/2018-EI-reg-En-anger-dev.txt", header=None, names=["text", "emotion", "intensity"], sep="\t")
df_anger_es_train = pd.read_csv("data/es/2018-EI-reg-Es-anger-train.txt", header=None, names=["text", "emotion", "intensity"], sep="\t")
df_anger_es_dev = pd.read_csv("data/es/2018-EI-reg-Es-anger-dev.txt", header=None, names=["text", "emotion", "intensity"], sep="\t")

df_anger_en = df_anger_en_train.append([df_anger_en_dev])
df_anger_es = df_anger_es_train.append([df_anger_es_dev])

df_anger_en["emotion"] = pd.Categorical(df_anger_en["emotion"]).codes
df_anger_es["emotion"] = pd.Categorical(df_anger_es["emotion"]).codes

X_en = df_anger_en["text"]
y_en = df_anger_en["intensity"]
X_es = df_anger_es["text"]
y_es = df_anger_es["intensity"]

df_anger_en.tail(10)

# Pipeline function
def use_pipeline(X, y, pipeline=None, estimator=None):
    if estimator:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)    
        estimator.fit(X_train, y_train)
        print("Classifier {}".format(str(pipeline.steps[2][1]).split("(")[0]))
        print("best R2 score: {}".format(estimator.best_score_))
        print("best parameters {}".format(estimator.best_params_))
        y_pred = estimator.predict(X_test)
        print("Pearson R coefficient {0}".format(pearsonr(y_test, y_pred)))
        print()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)    
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(pearsonr(y_test, y_pred))


clf_list = [LinearRegression(), Ridge(), RandomForestRegressor(), AdaBoostRegressor(), SVR()]
params = [{'vect__ngram_range': [(1,3), (1,2), (2,3)], 'vect__max_df': [0.8, 0.9, 1.0], 'vect__min_df': [2,3, 0.01], 'vect__max_features': [None, 100, 500, 1000, 2000, 5000, 8000], 'tfidf__norm': ['l1', 'l2', None], 'tfidf__use_idf': [True, False], 'tfidf__smooth_idf': [True, False], 'tfidf__sublinear_tf': [True, False], 'clf__normalize': [True, False]}, 
            {'vect__ngram_range': [(1,3), (1,2), (2,3)], 'vect__max_df': [0.8, 0.9, 1.0], 'vect__min_df': [2,3, 0.01], 'vect__max_features': [None, 100, 500, 1000, 2000, 5000, 8000], 'tfidf__norm': ['l1', 'l2', None], 'tfidf__use_idf': [True, False], 'tfidf__smooth_idf': [True, False], 'tfidf__sublinear_tf': [True, False], 'clf__alpha': [0.001, 0.01, 0.1, 1.0], 'clf__normalize': [True, False]}, 
            {'vect__ngram_range': [(1,3), (1,2), (2,3)], 'vect__max_df': [0.8, 0.9, 1.0], 'vect__min_df': [2,3, 0.01], 'vect__max_features': [None, 100, 500, 1000, 2000, 5000, 8000], 'tfidf__norm': ['l1', 'l2', None], 'tfidf__use_idf': [True, False], 'tfidf__smooth_idf': [True, False], 'tfidf__sublinear_tf': [True, False], 'clf__n_estimators': [5, 10, 20, 30, 40], 'clf__max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
             {'vect__ngram_range': [(1,3), (1,2), (2,3)], 'vect__max_df': [0.8, 0.9, 1.0], 'vect__min_df': [2,3, 0.01], 'vect__max_features': [None, 100, 500, 1000, 2000, 5000, 8000],'tfidf__norm': ['l1', 'l2', None], 'tfidf__use_idf': [True, False], 'tfidf__smooth_idf': [True, False], 'tfidf__sublinear_tf': [True, False], 'clf__learning_rate': [0.001, 0.01, 0.1, 1, 10], 'clf__n_estimators': [10, 30, 50, 70], 'clf__loss': ['linear', 'square', 'exponential']}, 
             {'vect__ngram_range': [(1,3), (1,2), (2,3)], 'vect__max_df': [0.8, 0.9, 1.0], 'vect__min_df': [2,3, 0.01], 'vect__max_features': [None, 100, 500, 1000, 2000, 5000, 8000], 'tfidf__norm': ['l1', 'l2', None], 'tfidf__use_idf': [True, False], 'tfidf__smooth_idf': [True, False], 'tfidf__sublinear_tf': [True, False], 'clf__C': [0.001, 0.01, 0.1, 1, 10], 'clf__kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'clf__shrinking': [True, False]}]

def get_best_params_per_classifier(X, y, clf_list, param_list):
    for index, clf in enumerate(clf_list):
        pipeline = Pipeline([
            ("vect", CountVectorizer(analyzer="word")),
            ("tfidf", TfidfTransformer(use_idf=True)),
            ('clf', clf)
        ])
        estimator = GridSearchCV(pipeline, param_list[index], scoring="r2")
        use_pipeline(X, y, pipeline, estimator=estimator)

print("-----------------------------")
print("classification results English")
get_best_params_per_classifier(X_en, y_en, clf_list, params)
print("-----------------------------")
print("classification results Spanish")
get_best_params_per_classifier(X_es, y_es, clf_list, params)




"""
# English word2vec pre-trained word embeddings
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)


# Spanish word2vec pre-trained word embeddings
model2 = gensim.models.KeyedVectors.load_word2vec_format('sbw_vectors.bin', binary=True)
model2.save_word2vec_format("sbw_vectors.txt", binary=False)


print(model.most_similar(positive=['woman', 'king'], negative=['man']))
print(model.similarity('woman', 'man'))

print(model2.most_similar(positive=['mujer', 'rey'], negative=['hombre']))
print(model2.similarity('mujer', 'hombre'))

"""
