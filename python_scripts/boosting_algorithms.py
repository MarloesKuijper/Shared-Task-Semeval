import os, sys, re, subprocess, shlex, argparse
import numpy as np
import csv
import pandas as pd
import random

## Sklearn

from sklearn import svm
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from random import uniform
from xgboost import XGBRegressor, XGBClassifier  ##https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/ , used by SeerNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings("ignore") #supress annoying sklearn warnings



""" Test a variety of different boosting algorithms """

### USAGE: python boosting_algorithms.py -f FOLDER [--clf] [--test TEST_FOLDER] [--ensemble] [--search]

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--features", required=True, type=str, help="Directory with all files with extracted features (vectors) OR a single file")
	parser.add_argument("-t","--test", default = '', type=str, help="Directory with all files with extracted features (vectors) for DEV/TEST OR a file -- if not added we do cross validation")
	parser.add_argument("-e","--extension", default = '.csv', type=str, help="Extension of feature vector files (default .csv)")
	parser.add_argument("-c","--clf", action = 'store_true', help="Select this if it is a classification task")
	parser.add_argument("-ens","--ensemble", action = 'store_true', help="Select this if we want to test an ensemble of algorithms")
	parser.add_argument("-s","--search", action = 'store_true', help="Select this if we want to a parameter search over a couple of algorithms")
	args = parser.parse_args()
	return args


def get_files(files, ext, options):
	""" returns files from directory (and subdirectories) 
		Only get files with certain extension""" 
	if os.path.isfile(files): #if it was a file, just return that in a list
		return [files], []
	else:
		file_list = []
		for path, subdirs, files in os.walk(files):
			for name in files:
				if name.endswith(ext):
					file_list.append(os.path.join(path, name))

		final_list = []
		emotion_order = [] #keep track of order of emotions so we dont have to break our head later to find it
		for em in options:
			for f in file_list:
				if em in f: #found emotion in file, add
					final_list.append(f)
					emotion_order.append(em)
					break	 #only add once per emotion
		return final_list, emotion_order


def get_datasets(f, feature_vectors_test, idx, clf, test):
	'''Get different datasets for training + testing -- use different ways for regression or clf data'''
	if clf:
		train_X, train_Y, options_train, old_options_train = load_clf_data(f)
	else:
		train_X, train_Y = load_reg_data(f)		

	## If there is a test file we get that, else send empty lists so function knows we want to do cross validation instead
	if test:
		if clf:
			test_X, test_Y, _,_ = load_clf_data(feature_vectors_test[idx])
		else:
			test_X, test_Y = load_reg_data(feature_vectors_test[idx])	
	else:
		test_X = []
		test_Y = []
		options_test = []	
	## For classification tasks we have to keep track of the number of options
	if args.clf:
		options = list(set(options_train))
		old_options = list(set(old_options_train))
	else:
		options = []
		old_options = []	
	
	return train_X, train_Y, test_X, test_Y, sorted(options), sorted(old_options)	


def rescale(Y, options):
	'''Rescale categories between 0 and 1'''
	sorted_options = sorted(options)
	range_divider = len(options) + 1
	new_options = []
	
	## Scale options between 0 and 1 evenly
	for idx, option in enumerate(options):
		new_val = round((float(1) / float(range_divider)) * (idx+1), 5)
		new_options.append(new_val)
	
	## Rewrite the vector by new options
	new_Y = []
	for y in Y:
		new_Y.append(new_options[sorted_options.index(y)])
	
	#print ('previous categories', sorted_options)
	#print ('current categories', new_options, '\n')
	return new_Y, new_options		
	

def load_clf_data(f):
	'''Load dataset for classification data'''
	dataset = pd.read_csv(f, skiprows=1)
	X = dataset.iloc[:,0:-1] #select everything but last column (label)
	Y = dataset.iloc[:,-1]   #select column
	Y, old_options = cat_to_int(Y)
	Y, options = rescale(Y, old_options)
	return X,Y, options, old_options

def load_reg_data(f):	
	'''Load dataset for regression data'''
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	X = dataset[:,0:-1] #select everything but last column (label)
	Y = dataset[:,-1]   #select column
	return X,Y


def is_empty(X):
	'''Check if panda/vector is empty'''
	return len(X) < 1

	
def cat_to_int(pred):
	'''Convert predicted categories to numbers'''
	new_pred = []
	options = []
	for p in pred:
		try:
			new_value = int(p[1]) 	#predicted category looks something like this: '0: no se infieren niveles de enojo' -- so take second character as number
		except ValueError:
			new_value = int(p[1:3]) #predicted category looks something like this: '-1: no se infieren niveles de enojo' -- so take second + third character as number
		new_pred.append(new_value)
		if new_value not in options:
			options.append(new_value)	
	
	return np.asarray(new_pred), options

		
def train_test_pearson(train_X, train_Y, test_X, test_Y, clf, clf_bool, options, old_options):
	'''Function that does fitting and pearson correlation with 10-fold cross validation'''
	
	## Do cross validation if test files are empty
	empty = is_empty(test_X)	
	if empty:
		res = cross_val_predict(clf, train_X, train_Y, cv=10, n_jobs=10) ##runs on 10 CPUs
		if clf_bool:
			num_res = res
			num_gold = train_Y
			num_res = [nearest_value(y, options, old_options)  for y in num_res] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
			return round(pearsonr(num_res, num_gold)[0],4), num_res
		else:
			return round(pearsonr(res, train_Y)[0],4), res	
	
	## Do testing with dev/test files and train on train
	else:
		pred = clf.fit(train_X, train_Y).predict(test_X)
		if clf_bool:
			num_res = pred
			num_gold = test_Y
			num_res = [nearest_value(y, options, old_options) for y in num_res] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
			return round(pearsonr(num_res, num_gold)[0],4), num_res
		else:	
			return round(pearsonr(pred, test_Y)[0],4), pred
	


def adaboost_search(train_X, train_Y, test_X, test_Y, estimators, learning_rates, emotion, res_dict, clf, options, old_options):
	'''Parameter search over adaboost. Can take quite a long time to finish'''
	print ('Testing AdaBoost...')
	for est in estimators:
		for rate in learning_rates:
			score, _ = train_test_pearson(train_X, train_Y, test_X, test_Y, AdaBoostRegressor(n_estimators=est, learning_rate=rate), clf, options, old_options)
			print ('Tested est={0} and learning_rate={1} with score {2}'.format(est, rate, score))	 
			res_dict['{0} - AdaBoost - est={1} - learning_rate={2} --> {3}\n'.format(emotion, est, rate, score)] = score #save res to dict
	
	return res_dict		


def xgboost_search(train_X, train_Y, test_X, test_Y, estimators, learning_rates, emotion, res_dict, clf, options, old_options):
	'''Parameter search over xg boost. Can take quite a long time to finish'''
	print ('Testing XGBoost...')
	for est in estimators:
		for rate in learning_rates:	
			score, _ = train_test_pearson(train_X, train_Y, test_X, test_Y, XGBRegressor(n_estimators=est, learning_rate=rate), clf, options, old_options)
			print ('Tested est={0} and learning_rate={1} with score {2}'.format(est, rate, score))	
			res_dict['{0} - XGBoost - est={1} - learning_rate={2} --> {3}\n'.format(emotion, est, rate, score)] = score #save res to dict
	
	return res_dict	


def forest_search(train_X, train_Y, test_X, test_Y, emotion, num_trees, res_dict, clf, options, old_options):
	'''Parameter search for random forests'''
	print ('Testing random forests')
	for num in num_trees:
		score, _ = train_test_pearson(train_X, train_Y, test_X, test_Y, RandomForestRegressor(n_estimators=num), clf, options, old_options)
		print ('Tested num_trees {0} with score {1}'.format(num, score))	 
		res_dict['{0} - RandomForest - num_trees={1} --> {2}\n'.format(emotion, num, score)] = score #save res to dict
	return res_dict	


def nearest_value(num, options, old_options):
	'''Get value in list of options which is closest to num'''
	nearest = min(options, key=lambda x:abs(x-num)) 	#nearest value scaled (so 0.25 or 0.5 or something)
	return_value =  old_options[options.index(nearest)]	#nearest value in original numbers	(so 1 or 2 or -2 etc)
	return return_value


def svm_search(train_X, train_Y, test_X, test_Y, epsilon, emotion, clf, options, old_options):
	'''SVM parameter search over epsilon OR over gamma/c'''
	best = [0,0]
	for eps in epsilon:
		svm_score, _ = train_test_pearson(train_X,train_Y, test_X, test_Y, svm.SVR(kernel='rbf', epsilon=eps), clf, options, old_options)
		if svm_score > best[0]:
			best = [svm_score, eps]
	print ('Best SVM score for eps: {0}: {1}'.format(best[1], best[0]))					
	return best


def print_best_score(res_dict, emotion):
	'''Print best score'''
	for w in sorted(res_dict, key=res_dict.get, reverse=True):
		print ('Best score: {0}'.format(res_dict[w]))
		break


def ensemble(train_X, train_Y, test_X, test_Y, clf, best_parameters, options, old_options):
	'''Ensemble  of (currently) 4 different regressors'''
	## First get individual scores
	svm_score, svm_pred = train_test_pearson(train_X,train_Y, test_X, test_Y, svm.SVR(kernel='rbf', epsilon=best_parameters[1]), clf, options, old_options)
	ada_score, ada_pred = train_test_pearson(train_X, train_Y, test_X, test_Y, AdaBoostRegressor(n_estimators=1000, learning_rate=0.1), clf, options, old_options)
	for_score, for_pred = train_test_pearson(train_X, train_Y, test_X, test_Y, RandomForestRegressor(n_estimators=1000), clf, options, old_options)
	xgb_score, xgb_pred = train_test_pearson(train_X, train_Y, test_X, test_Y, XGBRegressor(n_estimators=1000, learning_rate=0.1), clf, options, old_options)
	
	print ('Individual scores:', svm_score, ada_score, for_score, xgb_score)	
	## Final predictions are now the averages of the individual predictions (super simple ensemble)
	pred = np.asarray([float((svm_pred[idx] + ada_pred[idx] + for_pred[idx] + xgb_pred[idx])) / float(4) for idx in range(len(svm_pred))])
	
	## Put test_labels in correct format
	test_labels = train_Y if is_empty(test_Y) else test_Y
	ens_score = round(pearsonr(pred, test_labels)[0],4)
	print ('Ensemble score:', ens_score)


if __name__ == "__main__":
	args = create_arg_parser()
	
	options = ["anger", "fear", "joy", "sadness", "valence"]
		
	## Get files from directories
	feature_vectors_train, emotion_order = get_files(args.features, args.extension, options)
	if args.test:
		feature_vectors_test, emotion_order = get_files(args.test, args.extension, options)
		assert len(feature_vectors_test) == len(feature_vectors_train)
	else:
		feature_vectors_test = []	
	
	##Loop over feature vectors (emotions)
	for idx, f in enumerate(feature_vectors_train):
		emotion = 'single_file' if not emotion_order else emotion_order[idx] 		#get emotion
		print ('Testing {0}'.format(emotion))
		train_X, train_Y, test_X, test_Y, options, old_options = get_datasets(f, feature_vectors_test, idx, args.clf, args.test) #get data sets	

		## Always do SVM
		## For classification we always do a search because, default is not good
		epsilon = [0.001, 0.005, 0.35, 0.40] + [float(y * 0.01) for y in range(1,30)] #test very wide range of parameters here
		
		## Do SVM search here
		best_parameters = svm_search(train_X, train_Y, test_X, test_Y, epsilon, emotion, args.clf, options, old_options)
		
		## Try ensemble here if we want (doesn't seem to help though)
		if args.ensemble:
			ensemble(train_X, train_Y, test_X, test_Y, args.clf, best_parameters, options, old_options)
		
		## Do parameter search for SVM + 3 different boosting regressors currently, AdaBoost, XGBoost, RandomForrests
		if args.search:
			## Set parameters we want to search here
			res_dict = {}
			#estimators 		= [1500, 2000, 3000, 4000]
			#learning_rates 	= [0.01, 0.02, 0.05, 0.075, 0.1]
			estimators 		= [1500]
			learning_rates  = [0.02]
			
			## Now test the different algorithms
			res_dict = adaboost_search(train_X, train_Y, test_X, test_Y, estimators, learning_rates, emotion, res_dict, args.clf, options, old_options)
			res_dict = xgboost_search(train_X, train_Y, test_X, test_Y,estimators, learning_rates, emotion, res_dict, args.clf, options, old_options)
			res_dict = forest_search(train_X,train_Y, test_X, test_Y, emotion, estimators, res_dict, args.clf, options, old_options)

			# Get best score from dictionary and print
			print_best_score(res_dict, emotion)
		
