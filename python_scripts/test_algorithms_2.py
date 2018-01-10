import os, sys, re, subprocess, shlex, argparse
import numpy as np

## time
import time
import datetime

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
	parser.add_argument("-b","--bestalgorithms", default = '../best_algorithms/', type=str, help="Directory to store the algorithm results")
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


def train_test_pearson(X, Y, clf):
	'''Function that does fitting and pearson correlation
	   Note: added cross validation'''

	res = cross_val_predict(clf, X, Y, cv=10)
	return round(pearsonr(res, Y)[0],4)

def svc_param_selection(X, Y, clf):
	'''Same correlation test as train_test_pearson changed for
	   grid search to return best parameters'''

	# fit the classifier for retrieving the best parameters
	clf.fit(X,Y)

	# get best parameter
	best_param = clf.best_params_

	# get best estimator to use for prediction and correlation
	clf = clf.best_estimator_

	# do prediction with cross validation of 10
	res = cross_val_predict(clf, X, Y, cv=10)

	# return pearson correlation
	return round(pearsonr(res, Y)[0],4), clf

if __name__ == "__main__":

	# get time and date to name directory
	time_x = time.ctime()
	time_x = time_x.split()
	time_x = time_x[2] + time_x[1] + time_x[-1] + time_x[3]

	# create args parser
	args = create_arg_parser()

	## Get files from directories
	feature_vectors = get_files(args.features, args.extension)

	##Loop over different scores per file
	for f in feature_vectors:

		# start run timer
		t0 = time.time()

		## Try a few different regression algorithms -- taken from http://scikit-learn.org/stable/supervised_learning.html
		print ('Start testing different algorithms for {0}'.format(f))
		dataset = np.loadtxt(f, delimiter=",", skiprows = 1)

		## split into input (X) and output (Y) variables ##
		X = dataset[:,0:-1] #select everything but last column (label)
		Y = dataset[:,-1]   #select column

		## SVM test ##
		print("Running default SVM....")
		svm_score = train_test_pearson(X,Y, svm.SVR())

		## Ridge regression ##
		print("Running ridge regression....")
		ridge_score = train_test_pearson(X,Y, Ridge(alpha=1.0))

		## Lasso regression ##
		print("Running lasso regression....")
		lasso_score = train_test_pearson(X,Y, linear_model.Lasso(alpha=0.1))

		## KNN regression
		print("Running KNN regression....")
		knn_score = train_test_pearson(X,Y, KNeighborsRegressor(n_neighbors=4))

		## Decision tree
		print("Running decision tree....")
		tree_score = train_test_pearson(X,Y, tree.DecisionTreeRegressor())

		## Neural net MLP --lots of parameters here
		print("Running neural MLP regression....")
		mlp_score = train_test_pearson(X,Y, MLPRegressor())

		## Random forrest (currently commented out because it takes the longest)
		print("Running random forrest....")
		for_score = train_test_pearson(X,Y, RandomForestRegressor(n_estimators=200))
		#for_score = 0

		## Adaboost ensembler
		print("Running adaboost....")
		ada_score = train_test_pearson(X,Y, AdaBoostRegressor())

		## Gradient boost regressor
		print("Running gradient boost regression....")
		gra_score = train_test_pearson(X,Y, GradientBoostingRegressor())

		## If you want to do a parameter search you can do something like this
		print("Running KNN parameter search....")
		neighbours = [2,4,5,8,16,32]
		best_knn = -1000
		for n in neighbours:
			knn_score = train_test_pearson(X,Y, KNeighborsRegressor(n_neighbors=n))
			if knn_score > best_knn:
				best_knn = knn_score
				best_neighbours = n

		## grid search on SVM
		print("Running SVM grid search.....")
		parameters = {'c':[1, 10, 20, 30], 'epsilon':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],'cache_size':[300]}

		# dummy parameter for test
		#parameters = {'C':[0.8, 0.9]}

		# get best parameter and do pearson correlation in function svc_param_selection
		svm_search = svc_param_selection(X,Y, GridSearchCV(svm.SVR(kernel='rbf'), parameters))

		## set 0 to turn off grid search
		#svm_search = 0

		############################################################

		## Find best algorithm (kind of ugly but ok)
		all_scores = [svm_score, ridge_score, lasso_score, knn_score, tree_score, mlp_score, for_score, ada_score, gra_score, best_knn, svm_search[0]]
		all_ids = ['svm_default', 'ridge', 'lasso', 'knn', 'tree', 'mlp', 'forrests', 'adaboost', 'gradient', 'best_knn', 'best_svm']

		## process for naming directory based on date and time
		filename = f.split('/')
		feature_directory = str(filename[2])

		result_directory = filename[-1].split('.')[:-1]
		result_directory = result_directory[0]

		directory = str(args.bestalgorithms) + str(feature_directory) + '/' + str(result_directory) + '/' + str(time_x)

		## create dir if not exists
		if not os.path.exists(directory):
			os.makedirs(directory)

		# write results to txt file
		with open(directory + "/RESULTS.txt", "w") as outfile:

			for s,i in zip(all_scores, all_ids):
				print (i, s)
				text = i, s
				outfile.write(str(text))
				outfile.write("\n")

			outfile.write('\nBest algorithm: {0} with score {1}'.format(all_ids[all_scores.index(max(all_scores))], max(all_scores)))
			outfile.write('\nBest SVM parameters: {0} with score {1}'.format(svm_search[1], svm_search[0]))

			# end run timer
			run_time = time.time() - t0

			outfile.write('\n \nRun time in sec {0}'.format(run_time))

		## print best algorithm and run time in terminal
		print('\nBest algorithm: {0} with score {1}'.format(all_ids[all_scores.index(max(all_scores))], max(all_scores)))
		print("Run time in sec: ", run_time)
		print()
