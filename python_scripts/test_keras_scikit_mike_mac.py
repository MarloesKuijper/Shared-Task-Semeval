# coding=utf-8

'''
Script for testing Scikit and Keras
'''

# to run program in bash: python test_keras_scikit_mike.py -f1 ./features


import argparse
import sys
import ntpath
from importlib import reload
reload(sys)
# sys.setdefaultencoding('utf-8')    #necessary to avoid unicode errors
import os
import re
import numpy as np
from os import listdir
from os.path import isfile, join
# from os import walk
import pyexcel as p
import csv


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
## Keras

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

## Sklearn

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", required = True, type=str, help="Input folder")
    args = parser.parse_args()
    return args


def baseline_model(nodes, input_dim):
    # create model
    model = Sequential()
    model.add(Dense(nodes, input_dim = input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_test_pearson(clf, X_train, y_train, X_test, y_test):
    '''Function that does fitting and pearson correlation'''
    clf.fit(X_train, y_train)
    res = clf.predict(X_test)
    print("Pearson coefficient: {0}\n".format(pearsonr(res,y_test)[0]))

    return pearsonr(res, y_test)[0]


if __name__ == "__main__":
    args = create_arg_parser()

    # arg f1 is now a folder to get the files from (USE ./features/ OTHERWISE CHANGE lang and emotion below because they won't be split up properly!!!)
    filenames = []
    value = str(args.f1)
    for dirpath, dirnames, files in os.walk('../features'):
        print(dirnames)
        for name in files:
            if name not in ".DS_Store":
                print(name)
                filenames.append(os.path.join(dirpath, name))
    excel_data = []

    print("filenames:")
    print(filenames)

    for file in tqdm(filenames):
        ## load dataset ##
        task = "EI-REG"
        lang = file.split("/")[2].split("/")[-1]  # do this differently if you do not use -f1 = ./features
        print(lang)
        emotion = file.split("/")[3]  # do this differently if you do not use -f1 = ./features
        print(emotion)
        feat = file.split("/")[-1][:-4]
        print(feat)
        print(task, lang, emotion, feat)
        dataset = np.loadtxt(file, delimiter=",", skiprows = 1)

        ## split into input (X) and output (Y) variables ##
        X = dataset[:,0:-1] #select everything but last column (label)
        Y = dataset[:,-1]   #select column
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print("PREDICTIONS ", file)
        ## SVM test ##
        svm_clf = svm.SVR()
        print('Training SVM...\n')
        pearson_svm = train_test_pearson(svm_clf, X_train, y_train, X_test, y_test)

        ## Running baseline neural model ##
        print('Training neural baseline...\n')
        input_dim = len(X_train[0]) #input dimension is a necessary argument for the baseline model
        estimator = KerasRegressor(build_fn=baseline_model, nodes = 150, input_dim = input_dim, nb_epoch=100, batch_size=5, verbose=0)
        pearson_neural = train_test_pearson(estimator, X_train, y_train, X_test, y_test)
        found = False
        for item in excel_data:
            if task in item.values() and lang in item.values() and emotion in item.values():
                item[feat] = (float("{0:.2f}".format(pearson_svm)), float("{0:.2f}".format(pearson_neural)))
                found = True
        if not found:
            excel_data.append({"task": task, "lang": lang, "emotion": emotion, feat: (float("{0:.2f}".format(pearson_svm)), float("{0:.2f}".format(pearson_neural)))})
        print((float("{0:.2f}".format(pearson_svm)), float("{0:.2f}".format(pearson_neural))))


    keys = ["task", "lang", "emotion", "ngrams", "ngrams-embeddings", "ngrams-lexicons", "ngrams-lexicons-embeddings", "lexicons", "lexicons-embeddings", "embeddings"]
    with open('results_rounded_newest.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, delimiter=";")
        dict_writer.writeheader()
        for row in excel_data:
            dict_writer.writerow(row)



### TASK LANG EMOTION FEAT 1 FEAT 2 FEAT 3
