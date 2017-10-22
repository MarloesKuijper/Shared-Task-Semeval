# coding=utf-8

'''
Script for testing Scikit and Keras
'''


import argparse
import sys
from importlib import reload
reload(sys)
# sys.setdefaultencoding('utf-8')	#necessary to avoid unicode errors
import os
import re
import numpy as np
from os import listdir
from os.path import isfile, join
from os import walk

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


if __name__ == "__main__":
	args = create_arg_parser()

	# arg f1 is now a folder to get the files from (e.g. just ./features/en or ./features to get all languages)
	print(args)
	print(args.f1)
	filenames = []
	for (dirpath, dirnames, files) in walk(args.f1):
	    for name in files:
	    	filenames.append(os.path.join(dirpath, name))
	print(filenames)
	print("done")
	for file in filenames:
		## load dataset ##
		dataset = np.loadtxt(file, delimiter=",", skiprows = 1)
		
		## split into input (X) and output (Y) variables ##
		X = dataset[:,0:-1] #select everything but last column (label)
		Y = dataset[:,-1]   #select column
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
		
		print("PREDICTIONS ", file)
		## SVM test ##
		svm_clf = svm.SVR()
		print('Training SVM...\n')
		train_test_pearson(svm_clf, X_train, y_train, X_test, y_test)
		
		## Running baseline neural model ##
		print('Training neural baseline...\n')
		input_dim = len(X_train[0]) #input dimension is a necessary argument for the baseline model
		estimator = KerasRegressor(build_fn=baseline_model, nodes = 150, input_dim = input_dim, nb_epoch=100, batch_size=5, verbose=0)
		train_test_pearson(estimator, X_train, y_train, X_test, y_test)
