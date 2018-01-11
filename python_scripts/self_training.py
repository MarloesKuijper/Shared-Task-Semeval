import os, sys, re, subprocess, argparse
import numpy as np
from sklearn import svm
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict

""" Try self-learning: first train an algorithm on the training data --> test on unseen data and use that data as new training data"""

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1", required=True, type=str, help="Input-file gold (csv)")
	parser.add_argument("-d", required=True, type=str, help="Dev set -- so we can compare the impact of adding silver data (csv)")
	parser.add_argument("-s", required=True, type=str, help="Silver data to be used for self training (csv)")
	args = parser.parse_args()
	return args


def train_test_pearson(train_X, train_Y, test_X, test_Y, clf):
	'''Function that does fitting and pearson correlation with 10-fold cross validation'''
	
	## Do cross validation if test files are empty
	if test_X == [] or test_Y == []:
		res = cross_val_predict(clf, train_X, train_Y, cv=10, n_jobs=10) ##runs on 10 CPUs
		return round(pearsonr(res, train_Y)[0],4), res
	## Do testing with dev/test files and train on train
	else:
		pred = clf.fit(train_X, train_Y).predict(test_X)
		return round(pearsonr(pred, test_Y)[0],4), pred


def load_dataset(f):
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	X = dataset[:,0:-1] 
	Y = dataset[:,-1] 
	return X, Y  


if __name__ == "__main__":
	args = create_arg_parser()
	
	##Number of silver instances we add initially
	add_silver = 2000
	
	## Get scores from original dataset
	train_X, train_Y = load_dataset(args.f1)
	dev_X, dev_Y = load_dataset(args.d)
	silver_X, silver_Y = load_dataset(args.s)
	
	orig_score, orig_pred = train_test_pearson(train_X,train_Y, dev_X, dev_Y, svm.SVR(kernel='rbf'))
	print ('Original score: {0}'.format(orig_score))
	
	## Get prediction for silver dataset
	_, silver_pred = train_test_pearson(train_X,train_Y, silver_X, silver_Y, svm.SVR(kernel='rbf'))
	
	## Add silver data to gold data and print score
	add_X = np.concatenate((train_X, silver_X[0:add_silver]), axis=0)
	add_Y = np.concatenate((train_Y, silver_pred[0:add_silver]), axis=0)
	
	silver_score, pred_silver = train_test_pearson(add_X, add_Y, dev_X, dev_Y, svm.SVR(kernel='rbf'))
	print ('Silver score: {0}'.format(silver_score))
	
	##Get score for silver data ONLY
	silver_only_X = silver_X[0:add_silver]
	silver_only_Y = silver_pred[0:add_silver] 
	
	silver_only_score, _ = train_test_pearson(silver_only_X, silver_only_Y, dev_X, dev_Y, svm.SVR(kernel='rbf'))
	print ('Silver only score: {0}'.format(silver_only_score))
	
