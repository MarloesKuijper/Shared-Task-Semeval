import numpy as np 
from sklearn import svm 
import argparse
import os
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
import pandas as pd


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1","--feat_train", required=True, type=str, help="Features dir training data")
	parser.add_argument("-f2","--feat_dev", required=True, type=str, help="Features dir dev data")
	parser.add_argument("-f3","--feat_test", required=True, type=str, help="Features dir test data")
	parser.add_argument("-to","--original_test", required=True, type=str, help="Dir with original .txt test files")
	parser.add_argument("-d","--out_dir", required=True, type=str, help="Directory where you want the results to be written to")
	parser.add_argument("-c","--clf", action = 'store_true', help="Select this if it is a classification task")
	args = parser.parse_args()
	return args

def get_files(files, ext, options):
	""" returns files from directory (and subdirectories) 
		Only get files with certain extension, but if ext == 'all' we get everything """ 
	file_list = []
	for path, subdirs, files in os.walk(files):
		for name in files:
			if name.endswith(ext) or ext == 'all':
				file_list.append(os.path.join(path, name))
	
	## Now sort based on options so that all emotions are in same order
	final_list = []
	emotion_order = []
	for em in options:
		for f in file_list:
			if em in f.lower(): #found emotion in file, add
				final_list.append(f)
				emotion_order.append(em)
				break #only add once per emotion
	return final_list, emotion_order


def cat_to_int(pred):
	'''Convert predicted categories to numbers'''
	new_pred = []
	options = []
	for idx, p in enumerate(pred):
		try:
			new_value = int(p[1]) 	#predicted category looks something like this: '0: no se infieren niveles de enojo' -- so take second character as number
		except ValueError:
			new_value = int(p[1:3]) #predicted category looks something like this: '-1: no se infieren niveles de enojo' -- so take second + third character as number
		new_pred.append(new_value)
		if new_value not in options:
			options.append(new_value)	
	
	return np.asarray(new_pred), options


def get_datasets(f, clf):
	'''Get different datasets for training + testing -- use different ways for regression or clf data'''
	if clf:
		train_X, train_Y, options_train, old_options_train = load_clf_data(f, False)
	else:
		train_X, train_Y = load_reg_data(f)		
	
	## For classification tasks we have to keep track of the number of options
	if clf:
		options = list(set(options_train))
		old_options = list(set(old_options_train))
	else:
		options = []
		old_options = []	
	
	return train_X, train_Y, sorted(options), sorted(old_options)		


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
	return new_Y, new_options		
	

def load_clf_data(f, test):
	'''Load dataset for classification data'''
	# got very strange errors using pandas so did something else (super ugly but OK)
	#dataset = pd.read_csv(f, skiprows=1)
	#X = dataset.iloc[:,0:-1] #select everything but last column (label)
	
	dataset = []
	for line in open(f, 'r'):
		dataset.append(line.split(','))
	
	X = []
	Y = []
	for line in dataset[1:]:
		X.append(line[0:-1])
		Y.append(line[-1].strip())
	if not test:
		#Y = dataset.iloc[:,-1]   #select column
		Y, old_options = cat_to_int(Y)
		Y, options = rescale(Y, old_options)
		return X,Y, options, old_options
	else:
		return X

def load_reg_data(f):	
	'''Load dataset for regression data'''
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	X = dataset[:,0:-1] #select everything but last column (label)
	Y = dataset[:,-1]   #select column
	return X,Y


def train_test_pearson(train_X, train_Y, clf, clf_bool, options, old_options):
	'''Function that does fitting and pearson correlation with 10-fold cross validation'''
	
	## Do cross validation
	res = cross_val_predict(clf, train_X, train_Y, cv=10, n_jobs=10) ##runs on 10 CPUs
	if clf_bool:
		num_res = res
		num_gold = train_Y
		num_res = [nearest_value(y, options, old_options)  for y in num_res] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
		return round(pearsonr(num_res, num_gold)[0],4), num_res
	else:
		return round(pearsonr(res, train_Y)[0],4), res	


def svm_search(train_X, train_Y, epsilon, clf, options, old_options):
	'''SVM parameter search over epsilon'''
	best = [0,0]
	for eps in epsilon:
		svm_score, _ = train_test_pearson(train_X,train_Y, svm.SVR(kernel='rbf', epsilon=eps), clf, options, old_options)
		if svm_score > best[0]:
			best = [svm_score, eps]
	print ('Best SVM score for eps: {0}: {1}'.format(best[1], best[0]))					
	return best[0], best[1]


def nearest_value(num, options, old_options):
	'''Get value in list of options which is closest to num'''
	nearest = min(options, key=lambda x:abs(x-num)) 	#nearest value scaled (so 0.25 or 0.5 or something)
	return_value =  old_options[options.index(nearest)]	#nearest value in original numbers	(so 1 or 2 or -2 etc)
	return return_value


def predict_and_write_output(features_train, features_dev, features_test, original_test_file, out_dir, emotion):
	""" Adds train + dev together. Then does CV search for best parameters -- those are used on test set"""
	
	train_X, train_Y, options, old_options = get_datasets(features_train, args.clf)
	dev_X, dev_Y, options, old_options = get_datasets(features_dev, args.clf)
	train_dev_X = np.concatenate((train_X, dev_X), axis=0)
	train_dev_Y = np.concatenate((train_Y, dev_Y), axis=0)
	
	## Find best epsilon value for cross validation
	epsilon = [0.001, 0.005, 0.35, 0.40] #+ [float(y * 0.01) for y in range(1,30)] #test very wide range of parameters here
	best_score, best_eps = svm_search(train_X, train_Y, epsilon, args.clf, options, old_options)
	
	## Get test set (without labels)
	if args.clf:
		test_X = load_clf_data(features_test, True)
	else:
		test_X, _ = load_reg_data(features_test)	
	## Use this value to fit model to use on test set
	clf = svm.SVR(epsilon=best_eps, kernel='rbf')
	test_pred = clf.fit(train_dev_X, train_dev_Y).predict(test_X)
	if args.clf: ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
		test_pred = [nearest_value(y, options, old_options) for y in test_pred] 
	
	## Write predictions to file
	out_dir_full = "{0}{1}/".format(out_dir, emotion)
	if not os.path.exists(out_dir_full):
		os.system("mkdir -p {0}".format(out_dir_full))
	name = "{0}{1}/{1}_svm_search_pred.txt".format(out_dir, emotion)
	
	with open(original_test_file, 'r', encoding="utf-8") as infile:
		infile = infile.readlines()
		infile = [x for x in infile if x]
		data = ["\t".join(line.split("\t")[:-1]) + "\t" + str(test_pred[ix]) for ix, line in enumerate(infile[1:])]
		with open(name, 'w', encoding="utf-8") as out:
			out.write(infile[0])
			for line in data:
				out.write(line)
				out.write("\n")
		out.close()			
	return best_score, best_eps			


if __name__ == "__main__":
	args = create_arg_parser()
	train_dir 	  = args.feat_train
	dev_dir 	  = args.feat_dev
	test_dir 	  = args.feat_test
	original_test = args.original_test
	out_dir	 	  = args.out_dir
	
	if not os.path.exists(out_dir):
		os.system("mkdir -p {0}".format(out_dir))
	
	options = ["anger", "fear", "joy", "sadness", "valence"]

	training_feats, emotion_order = get_files(train_dir, ".csv", options)
	dev_feats, _ = get_files(dev_dir, ".csv", emotion_order)
	test_feats, _ = get_files(test_dir, ".csv", emotion_order)
	original_txts, _ = get_files(original_test, ".txt", emotion_order)
	
	assert len(training_feats) == len(dev_feats) == len(test_feats) == len(original_txts) #lengths must be the same
	
	general_file = '{0}svm_search_results.txt'.format(out_dir)
	with open(general_file, 'w') as out_file:
		for ix, file in enumerate(training_feats):
			print (training_feats[ix])
			print (test_feats[ix],'\n')
			best_score, best_eps = predict_and_write_output(training_feats[ix], dev_feats[ix], test_feats[ix], original_txts[ix], out_dir, emotion_order[ix])
			
			out_file.write('{0}: eps {1} and score {2}\n'.format(emotion_order[ix], best_eps, best_score))
	out_file.close()		
