import os, sys, re, subprocess, shlex, argparse
import numpy as np

## Sklearn

from sklearn import svm
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from random import uniform

""" Test a variety of different regression algorithms on a number of datasets"""

### USAGE: python train_test_pearsons.py -f FOLDER


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--features", required=True, type=str, help="Directory with all files with extracted features (vectors)")
	parser.add_argument("-e","--extension", default = '.csv', type=str, help="Extension of feature vector files (default .csv)")
	args = parser.parse_args()
	return args


def get_files(files, ext):
	""" returns files from directory (and subdirectories) 
		Only get files with certain extension""" 
	file_list = []
	for path, subdirs, files in os.walk(files):
		for name in files:
			if name.endswith(ext):
				file_list.append(os.path.join(path, name))
	return file_list


def train_test_pearson(X, Y, clf):
	'''Function that does fitting and pearson correlation
	   Note: added cross validation'''

	res = cross_val_predict(clf, X, Y, cv=10)
	return round(pearsonr(res, Y)[0],4)


if __name__ == "__main__":
	args = create_arg_parser()
	
	## Get files from directories
	feature_vectors = get_files(args.features, args.extension)
	
	##Loop over different scores per file
	for f in feature_vectors:
		## Try a few different regression algorithms -- taken from http://scikit-learn.org/stable/supervised_learning.html
		print ('Start testing different algorithms for {0}'.format(f))
		dataset = np.loadtxt(f, delimiter=",", skiprows = 1)

		## split into input (X) and output (Y) variables ##
		X = dataset[:,0:-1] #select everything but last column (label)
		Y = dataset[:,-1]   #select column
		
		## SVM test ##
		svm_score = train_test_pearson(X,Y, svm.SVR())
		
		## Ridge regression ##
		ridge_score = train_test_pearson(X,Y, Ridge(alpha=1.0))

		## Lasso regression ##
		lasso_score = train_test_pearson(X,Y, linear_model.Lasso(alpha=0.1))

		## KNN regression
		knn_score = train_test_pearson(X,Y, KNeighborsRegressor(n_neighbors=4))

		## Decision tree
		tree_score = train_test_pearson(X,Y, tree.DecisionTreeRegressor())

		## Neural net MLP --lots of parameters here
		mlp_score = train_test_pearson(X,Y, MLPRegressor())

		## Random forrest (currently commented out because it takes the longest)
		#for_score = train_test_pearson(X,Y, RandomForestRegressor(n_estimators=200))
		for_score = 0
		
		## Adaboost ensembler
		ada_score = train_test_pearson(X,Y, AdaBoostRegressor())

		## Gradient boost regressor
		gra_score = train_test_pearson(X,Y, GradientBoostingRegressor())

		## If you want to do a parameter search you can do something like this
		neighbours = [2,4,8,16]
		best_knn = -1000
		for n in neighbours:
			knn_score = train_test_pearson(X,Y, KNeighborsRegressor(n_neighbors=n))
			if knn_score > best_knn:
				best_knn = knn_score
				best_neighbours = n
		
		## You can also do grid searches over parameter settings automatically for e.g. SVMs
		## Parameters is a dictionary with the parameter as key and the settings as the value of the key in a list
		## Currently only few parameters so that it is still reaonsably fast, but can be expanded (commented out but works)

		#parameters = {'kernel':('linear', 'rbf'), 'C':[5], 'epsilon':[0.05]}
		#svm_search = train_test_pearson(X,Y, GridSearchCV(svm.SVR(), parameters)) 
		svm_search = 0
		
		## Find best algorithm (kind of ugly but ok)
		all_scores = [svm_score, ridge_score, lasso_score, knn_score, tree_score, mlp_score, for_score, ada_score, gra_score, best_knn, svm_search]
		all_ids = ['svm_default', 'ridge', 'lasso', 'knn', 'tree', 'mlp', 'forrests', 'adaboost', 'gradient', 'best_knn', 'best_svm']
		
		for s,i in zip(all_scores, all_ids):
			print (i, s)
		
		print ('\nBest algorithm: {0} with score {1}'.format(all_ids[all_scores.index(max(all_scores))], max(all_scores)))
