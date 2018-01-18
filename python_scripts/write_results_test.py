import numpy as np 
from sklearn import svm 
import argparse
import os
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
import pandas as pd


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1","--main_dir", required=True, type=str, help="Main directory with train/dev/test/trans data (csv features)")
	parser.add_argument("-to","--original_test", required=True, type=str, help="Dir with original .txt test files")
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
		return np.asarray(X), np.asarray(Y), options, old_options
	else:
		return np.asarray(X)

def load_reg_data(f):	
	'''Load dataset for regression data'''
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	X = dataset[:,0:-1] #select everything but last column (label)
	Y = dataset[:,-1]   #select column
	return X,Y


def train_test_pearson(train_X, train_Y, dev_X, dev_Y, clf, clf_bool, options, old_options):
	'''Function that does fitting and pearson correlation with 10-fold cross validation'''
	
	## Do tests on dev files
	if len(dev_X) > 0:
		pred = clf.fit(train_X, train_Y).predict(dev_X)
		if clf_bool:
			num_res = pred
			num_gold = dev_Y
			num_res = [nearest_value(y, options, old_options) for y in num_res] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
			return round(pearsonr(num_res, num_gold)[0],4), num_res
		else:	
			return round(pearsonr(pred, dev_Y)[0],4), pred
	## Do cross validation
	else:	
		res = cross_val_predict(clf, train_X, train_Y, cv=10, n_jobs=10) ##runs on 10 CPUs
		if clf_bool:
			num_res = res
			num_gold = train_Y
			num_res = [nearest_value(y, options, old_options)  for y in num_res] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
			return round(pearsonr(num_res, num_gold)[0],4), num_res
		else:
			return round(pearsonr(res, train_Y)[0],4), res	


def svm_search(train_X, train_Y, dev_X, dev_Y, epsilon, clf, options, old_options):
	'''SVM parameter search over epsilon'''
	best = [0,0]
	scores = []
	for eps in epsilon:
		svm_score, svm_pred = train_test_pearson(train_X,train_Y, dev_X, dev_Y, svm.SVR(kernel='rbf', epsilon=eps), clf, options, old_options)
		scores.append(svm_score)
		if svm_score > best[0]:
			best = [svm_score, eps, svm_pred]					
	return best[0], best[1], best[2]


def nearest_value(num, options, old_options):
	'''Get value in list of options which is closest to num'''
	nearest = min(options, key=lambda x:abs(x-num)) 	#nearest value scaled (so 0.25 or 0.5 or something)
	return_value =  old_options[options.index(nearest)]	#nearest value in original numbers	(so 1 or 2 or -2 etc)
	return return_value


def cv_dataset(train_X, train_Y,  low, up):
	'''Apparently Keras has no cross validation functionality built in so do it here
	   The rows between low and up are for the validation set, rest is for train set'''
	return_train_X = np.concatenate((train_X[:low, :], train_X[up:, :] ), axis=0)
	return_valid_X = train_X[low:up, :]
	return_train_Y = np.concatenate((train_Y[:low], train_Y[up:] ), axis=0)
	return_valid_Y = train_Y[low:up]
	return return_train_X, return_train_Y, return_valid_X, return_valid_Y


def cross_val_svm(train_X, train_Y ,trans_X, trans_Y, best_eps_train, clf, options, old_options, folds):
	'''Have to do our own cross validation for SVM since we want to add the translated data ONLY to train'''
	all_preds = np.asarray([])
	all_scores = []
	for fold in range(folds):
		## Get datasets
		low = int(len(train_X) * fold / folds)
		up  = int(len(train_X) * (fold +1) / folds)
		new_train_X, new_train_Y, new_valid_X, new_valid_Y = cv_dataset(train_X, train_Y, low, up)
		final_train_X = np.concatenate((new_train_X, trans_X), axis=0)
		final_train_Y = np.concatenate((new_train_Y, trans_Y), axis=0)

		## Get predictions + score
		clf = svm.SVR(epsilon=best_eps_train, kernel='rbf')
		test_pred = clf.fit(final_train_X, final_train_Y).predict(new_valid_X)
		
		## Save predictions + correct score
		if args.clf: ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
			test_pred = [nearest_value(y, options, old_options) for y in test_pred] 
		all_preds = np.concatenate((all_preds, test_pred), axis=0)
		score = round(pearsonr(test_pred, new_valid_Y)[0],4)
		all_scores.append(score)
	
	return round(float(sum(all_scores)) / float(len(all_scores)),4), np.asarray(all_preds)

		
def test_and_write_output(train_X, train_Y, test_X, epsilon, options, old_options, out_dir, emotion, original_test_file, ident):
	
	## Use this value to fit model to use on test set
	clf = svm.SVR(epsilon=epsilon, kernel='rbf')
	test_pred = clf.fit(train_X, train_Y).predict(test_X)
	if args.clf: ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
		test_pred = [nearest_value(y, options, old_options) for y in test_pred] 
	
	# Write predictions to file
	out_dir_full = "{0}{1}/".format(out_dir, emotion)
	if not os.path.exists(out_dir_full):
		os.system("mkdir -p {0}".format(out_dir_full))
	name = "{0}{1}/{1}_{2}_svm_search_pred.txt".format(out_dir, emotion, ident)
	
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


def predict_and_write_output(features_train, features_dev, features_test, features_trans, original_test_file, out_dir, emotion, ens_dir):
	""" Adds train + dev together. Then does CV search for best parameters -- those are used on test set"""
	train_X, train_Y, options, old_options = get_datasets(features_train, args.clf)
	dev_X, dev_Y, options, old_options = get_datasets(features_dev, args.clf)
	trans_X, trans_Y, options, old_options = get_datasets(features_trans, args.clf)
	train_trans_X = np.concatenate((train_X, trans_X), axis=0)
	train_trans_Y = np.concatenate((train_Y, trans_Y), axis=0)
	
	## Find best epsilon value for cross validation
	epsilon = [0.001, 0.005, 0.35, 0.40] + [float(y * 0.01) for y in range(1,30)] #test very wide range of parameters here
	#epsilon = [0.02]
	
	## Find best parameters for: train + CV
	best_score_train, best_eps_train, pred_train  = svm_search(train_X, train_Y, [], [], epsilon, args.clf, options, old_options)
	
	## Apply parameter to get scores for dev set for train and train + translated
	best_score_train_trans, pred_train_trans = cross_val_svm(train_X, train_Y, trans_X, trans_Y, best_eps_train, args.clf, options, old_options, 10)
	best_score_dev, _, pred_dev   = svm_search(train_X, train_Y, dev_X, dev_Y, [best_eps_train], args.clf, options, old_options)
	best_score_dev_trans, _, pred_dev_trans = svm_search(train_trans_X, train_trans_Y, dev_X, dev_Y, [best_eps_train], args.clf, options, old_options)
	
	
	prt_str = "Train 10-fold CV (eps {0}) score: {1}\nDev-score: {2}\nTrain + trans on dev: {3}\nTrain + trans on 10-fold CV: {4}".format(best_eps_train, best_score_train, best_score_dev, best_score_dev_trans, best_score_train_trans)
	print (prt_str)
	
	## Get test set (without labels)
	if args.clf:
		test_X = load_clf_data(features_test, True)
	else:
		test_X, _ = load_reg_data(features_test)	
	
	## Write to file in correct format
	test_and_write_output(train_X, train_Y, test_X, best_eps_train, options, old_options, out_dir, emotion, original_test_file, 'traindev')
	test_and_write_output(train_trans_X, train_trans_Y, test_X, best_eps_train, options, old_options, out_dir, emotion, original_test_file, 'trans')
	
	## Write to files because we want to train ensembles later
	ens_dir_train = "{0}/{1}/train/".format(ens_dir, emotion)
	ens_dir_dev   = "{0}/{1}/dev/".format(ens_dir, emotion)
	
	write_ens_file(ens_dir_train , 'svm_train.txt', pred_train)
	write_ens_file(ens_dir_train , 'svm_train_trans.txt', pred_train_trans)
	write_ens_file(ens_dir_dev , 'svm_dev.txt', pred_dev)
	write_ens_file(ens_dir_dev , 'svm_dev_trans.txt', pred_dev_trans)
	
	return prt_str


def write_ens_file(f, name, pred):
	'''Write to ensemble file -- prediction only'''
	if not os.path.exists(f):
		os.system("mkdir -p {0}".format(f))
	f_name = f + name
	with open(f_name,'w') as out_f:
		for p in pred:
			out_f.write(str(p) +'\n')
	out_f.close()		


if __name__ == "__main__":
	args = create_arg_parser()
	train_dir 	  = args.main_dir + 'train/'
	dev_dir 	  = args.main_dir + 'dev/'
	test_dir 	  = args.main_dir + 'test/'
	trans_dir	  = args.main_dir + 'translated/'
	ens_dir	 	  = args.main_dir + 'ensemble/'
	out_dir	 	  = args.main_dir + 'predictions/'
	original_test = args.original_test
	
	if not os.path.exists(out_dir):
		os.system("mkdir -p {0}".format(out_dir))
	
	options = ["anger", "fear", "joy", "sadness", "valence"]

	training_feats, emotion_order = get_files(train_dir, ".csv", options)
	dev_feats, _ = get_files(dev_dir, ".csv", emotion_order)
	test_feats, _ = get_files(test_dir, ".csv", emotion_order)
	trans_feats, _ = get_files(trans_dir, ".csv", emotion_order)
	original_txts, _ = get_files(original_test, ".txt", emotion_order)
	
	#print(len(training_feats), len(dev_feats), len(test_feats) , len(original_txts))
	assert len(training_feats) == len(dev_feats) == len(test_feats) == len(trans_feats) == len(original_txts) #lengths must be the same
	
	general_file = '{0}svm_search_results.txt'.format(out_dir)
	with open(general_file, 'w') as out_file:
		for ix, file in enumerate(training_feats):
			print("For {0}".format(emotion_order[ix]))
			out_file.write("For {0}\n".format(emotion_order[ix]))
			prt_str = predict_and_write_output(training_feats[ix], dev_feats[ix], test_feats[ix], trans_feats[ix], original_txts[ix], out_dir, emotion_order[ix], ens_dir)
			out_file.write(prt_str + '\n\n')
	out_file.close()		
