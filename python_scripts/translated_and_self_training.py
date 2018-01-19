import argparse
import keras
import os
import numpy as np
from keras.callbacks import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from scipy.stats import pearsonr
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time
from keras import backend
from random import shuffle

""" Script that uses translated data to improve training + self-training with silver data to improve training """

#np.random.seed(42)

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1", required=True, type=str, help="Input-files gold (csv)")
	parser.add_argument("-d", required=True, type=str, help="Dev sets -- so we can compare the impact of adding silver data (csv)")
	parser.add_argument("-s", required=True, type=str, help="Silver data to be used for self training (csv)")
	parser.add_argument("-t", required=True, type=str, help="Translated data to be added to gold data")
	parser.add_argument("-te", required=True, type=str, help="To be tested data")
	parser.add_argument("-dor", required=True, type=str, help="Original files with dev data so we can write them to file in correct format")
	parser.add_argument("-tor", required=True, type=str, help="Original files with test data so we can write them to file in correct format")
	parser.add_argument("-o", required=True, type=str, help="Output folder")
	parser.add_argument("-r", required=True, type=str, help="Result file to keep track of all results for the different dev sets")
	parser.add_argument("-clf", action='store_true', help="If added this is a classification task")
	parser.add_argument("-lstm", action='store_true', help="If added we train an LSTM instead of a feed-forward network")
	parser.add_argument("-shuffle", action='store_true', help="If added we shuffle the rows before training (different results for cross validation)")
	args = parser.parse_args()
	return args
	

def nearest_value(num, options, old_options):
	'''Get value in list of options which is closest to num'''
	nearest = min(options, key=lambda x:abs(x-num)) 	#nearest value scaled (so 0.25 or 0.5 or something)
	return_value =  old_options[options.index(nearest)]	#nearest value in original numbers	(so 1 or 2 or -2 etc)
	return return_value


def cat_to_int(pred):
	'''Convert predicted categories to numbers'''
	new_pred = []
	options = []
	num_dict = {}
	for idx, p in enumerate(pred):
		#print (p)
		try:
			new_value = int(p[1]) 	#predicted category looks something like this: '0: no se infieren niveles de enojo' -- so take second character as number
		except ValueError:
			new_value = int(p[1:3]) #predicted category looks something like this: '-1: no se infieren niveles de enojo' -- so take second + third character as number
		new_pred.append(new_value)
		if new_value not in options:
			options.append(new_value)
		if new_value not in num_dict:
			num_dict[new_value] = p		
	
	return np.asarray(new_pred), options, num_dict


def get_dataset(f, clf, shuffle):
	'''Get dataset-- use different ways for regression or clf data'''
	if clf:
		train_X, train_Y, dataset, options_train, old_options_train, num_dict = load_clf_data(f, shuffle)
	else:
		train_X, train_Y, dataset = load_reg_data(f, shuffle)		
	
	## For classification tasks we have to keep track of the number of options
	if clf:
		options = list(set(options_train))
		old_options = list(set(old_options_train))
	else:
		options = []
		old_options = []
		num_dict = {}	
	
	return train_X, train_Y, dataset, sorted(options), sorted(old_options), num_dict		


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
	

def load_clf_data(f, shuffle_bool):
	'''Load dataset for classification data'''
	# got very strange errors using pandas so did something else (super ugly but OK)	
	
	## Get dataset and maybe shuffle
	dataset = []
	for line in open(f, 'r'):
		dataset.append(line.split(','))
	if shuffle_bool:
		shuffle(dataset)
	
	## Get X and Y
	X = []
	Y = []
	for line in dataset[1:]:
		X.append(line[0:-1])
		Y.append(line[-1].strip())
	
	## Convert to int and rescale back
	Y, old_options, num_dict = cat_to_int(Y)
	Y, options = rescale(Y, old_options)
	return np.asarray(X),np.asarray(Y), np.asarray(dataset), options, old_options, num_dict


def load_reg_data(f, shuffle):	
	'''Load dataset for regression data'''
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	if shuffle: #shuffle for more randomization
		np.random.shuffle(dataset)
	X = dataset[:,0:-1] 
	Y = dataset[:,-1]   #select column
	return X,Y, dataset


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
				if em in f.lower(): #found emotion in file, add
					final_list.append(f)
					emotion_order.append(em)
					break	 #only add once per emotion
		return final_list, emotion_order


def train_lstm(train_X, train_Y, dev_X, dev_Y, test_X,  input_dim, dense_layers, lstm_layers, nodes, dropout, clf, options, old_options, neurons):
	# don't use -scale here
	## Reshape data
	train_X = np.asarray(train_X)
	train_Y = np.asarray(train_Y)
	train_X = train_X.reshape((len(train_X), 1, len(train_X[0])))
	
	if len(dev_X) > 0 and len(dev_Y) > 0:
		dev_X = np.asarray(dev_X)
		dev_Y = np.asarray(dev_Y)
		dev_X = dev_X.reshape((len(dev_X), 1, len(dev_X[0])))
	
	if len(silver_X) > 0:
		silver_X = np.asarray(silver_X)
		silver_X = silver_X.reshape((len(silver_X), 1, len(silver_X[0])))
	#print (train_X.shape, train_Y.shape, dev_X.shape, dev_Y.shape)
	
	## Get model
	model = Sequential()
	
	## Add LSTM layers
	model.add(LSTM(neurons, input_shape=(len(train_X[0]), len(train_X[0][0])), dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation="relu"))  
	if lstm_layers == 2:
		model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=dropout, activation="relu")) 
	else: 
		model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation="relu")) 
		model.add(LSTM(int(nodes/2), dropout=dropout, recurrent_dropout=dropout, activation="relu"))
	
	## Add dense layers
	if dense_layers > 0:
		if lstm_layers == 2:
			model.add(Dense(int(nodes/2), activation="relu"))
			if dense_layers > 1:
				model.add(Dense(int(nodes/4), activation="relu"))
		else:
			model.add(Dense(int(nodes/4), activation="relu"))
			if dense_layers > 1:
				model.add(Dense(int(nodes/8), activation="relu"))
	
	## Final layer
	model.add(Dense(1, activation='sigmoid'))                       
	model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
	
	## Callback (stop training when we encounter NaN)
	callbacks = TerminateOnNaN()
	
	## Train model
	model.fit(train_X, train_Y, batch_size=16, epochs=20, callbacks=[callbacks], validation_split = 0.1, verbose=0)
	
	## Make predictions and evaluate, either on silver or test, and on dev (for silver dev is not important though)
	predictions_dev, score_dev, predictions_test = make_predictions (model, dev_X, dev_Y, test_X, options, old_options, clf)
	
	return predictions_dev, score_dev, predictions_test


def train_feedforward(train_X, train_Y, valid_X, valid_Y, test_X,  input_dim, dense_layer, last_layer, num_dense_layer, dropout, clf, options, options_old):
	'''Feedforward network similar to Approach 1 of the winners of the WASSA emotion intensity shared task
	   http://www.aclweb.org/anthology/W17-5207'''
	## Create model
	model = Sequential()
	model.add(Dense(dense_layer, input_dim=input_dim, activation="relu"))
	model.add(Dropout(dropout))
	for i in range(num_dense_layer):
		model.add(Dense(dense_layer, activation="relu"))
	model.add(Dense(last_layer, activation="relu"))
	model.add(Dense(1, activation="sigmoid"))

	## Train
	model.compile(loss='mse', optimizer='adam', metrics=['mse']) 	#mse as loss and metric, should be good enough even though it's not pearson
	model.fit(train_X, train_Y, batch_size=16, epochs=20, validation_split = 0.1, verbose=0)
	
	## Make predictions and evaluate, either on silver or test, and on dev (for silver dev is not important though)
	predictions_dev, score_dev, predictions_test = make_predictions (model, dev_X, dev_Y, test_X, options, options_old, clf)
	return predictions_dev, score_dev, predictions_test


def make_predictions (model, dev_X, dev_Y, test_X, options, old_options, clf):
	'''Make predictions on dev and test set with model'''
	
	if len(dev_X) > 0 and len(dev_Y) > 0:
		## Do dev predictions + score
		pred_dev = model.predict(dev_X, batch_size=8, verbose=0)
		dev_predictions = [p[0] for p in pred_dev] #put in format we can evaluate, avoid numpy error
		if clf:
			dev_predictions_score = [nearest_value(y, options, old_options) for y in dev_predictions] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
			dev_score = round(pearsonr(dev_predictions_score, dev_Y)[0], 4)
		else:
			dev_score = round(pearsonr(dev_predictions, dev_Y)[0], 4)
	else:
		dev_score, dev_predictions = _, _	
	
	## Do test predictions (no score because it is not gold)
	test_pred = model.predict(test_X, batch_size=8, verbose=0)
	test_predictions = [p[0] for p in test_pred] #put in format we can evaluate, avoid numpy error
	#if clf:
	#	test_predictions = [nearest_value(y, options, old_options)  for y in test_predictions] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
	return dev_predictions, dev_score, test_predictions


def cv_dataset(dataset, low, up):
	'''Apparently Keras has no cross validation functionality built in so do it here
	   The rows between low and up are for the validation set, rest is for train set'''
	train_rows = np.concatenate((dataset[:low, :], dataset[up:, :] ), axis=0)
	valid_rows = dataset[low:up, :]
	train_X = train_rows[:,0:-1] 
	train_Y = train_rows[:,-1] 
	valid_X = valid_rows[:,0:-1]
	valid_Y = valid_rows[:,-1] 
	return train_X, train_Y, valid_X, valid_Y


def average(l):
	return round((float(sum(l)) / float(len(l))),4)


def get_averages(predictions, silver_X, similarity_max):
	'''Take average of list of predictions, but only if min(l) and max(l) do not differ by more than similarity max'''
	new_silver_X = []
	new_avg = []
	
	for p in range(len(predictions[0])):
		avges = []
		for l in range(len(predictions)):
			avges.append(predictions[l][p])
		diff = max(avges) - min(avges)  #difference between minimum and maximum predictions
		if diff < similarity_max: 		#all predictions are similar enough to keep
			new_silver_X.append(silver_X[p])
			new_avg.append(average(avges))
	return new_silver_X, new_avg			

def get_silver_data(train_file, dev_file, silver_file, shuffle, runs, similarity_max, arg1, arg2, arg3, arg4, clf, out_file, lstm_bool):
	'''Get predictions for silver data -- first take average over 10 models and only take predictions that are similar enough to SVM
	   Arg1-4 are dependent on whether we use LSTM or feed-forward'''
	silver_X, silver_Y, _,_ ,_, _ = get_dataset(silver_file, False, True)
	
	predictions = []
	predictions_rounded = []
	for run in range(runs):
		train_X, train_Y, _ , options, options_old, _ = get_dataset(train_file, clf, shuffle)
		dev_X, dev_Y, _, _, _, _ = get_dataset(dev_file, clf, shuffle)
		
		## Get predictions on silver data set
		if lstm_bool:
			_, _, silver_pred = train_lstm(train_X, train_Y, [], [], silver_X, len(train_X[0]), arg1, arg2, arg3, arg4, clf, options, options_old, len(dev_Y))
		else:	
			_, _, silver_pred = train_feedforward(train_X, train_Y, [], [], silver_X,  len(train_X[0]), arg1, arg2, arg3, arg4, clf, options, options_old)
		predictions.append(silver_pred)
	
	## Simply average predictions
	average_predictions = np.mean(np.asarray(predictions), axis=0)
	
	## Also possible to only keep averages for which the predictions were similar enough
	prev_len = len(silver_X)
	silver_X, average_predictions = get_averages(predictions, silver_X, similarity_max)
	print_both('{0}: keep {1} out of {2}'.format(similarity_max, len(silver_X), prev_len), out_file)
	return silver_X, average_predictions


def get_task(f, lst):
	for l in lst:
		if l in f:
			return l
	raise ValueError("Could not find task: {0} not in {1}".format(lst, f))		


def print_both(string, f):
	'''Function to print both screen and to file'''
	print (string)
	if f:
		f.write(string + '\n')


def get_original_score(full_train_X, full_train_Y, dev_X, dev_Y, test_X, input_dim, arg1, arg2, arg3, arg4, clf, options, options_old, runs, lstm, out_file):
	'''Get original score when training on train set and testing on dev set. Do average over X runs to be sure'''
	dev_scores = []
	dev_preds = []
	test_preds = []
	for run in range(runs):
		if lstm:
			dev_pred, dev_score, test_pred = train_lstm(full_train_X, full_train_Y, dev_X, dev_Y, test_X, input_dim, arg1, arg2, arg3, arg4, args.clf, options, options_old, len(dev_Y))
		else:
			dev_pred, dev_score, test_pred = train_feedforward(full_train_X, full_train_Y, dev_X, dev_Y, test_X,  input_dim, arg1, arg2, arg3, arg4, args.clf, options, options_old)
		dev_scores.append(dev_score)
		dev_preds.append(dev_pred)
		test_preds.append(test_pred)
		
	## Keep track of results and print
	avg_score_dev, min_score_dev, max_score_dev = average(dev_scores), min(dev_scores), max(dev_scores)
	avg_preds_dev = np.asarray(dev_preds).mean(axis=0)
	avg_preds_test = np.asarray(test_preds).mean(axis=0)
	dev_pred_score = round(pearsonr(avg_preds_dev, dev_Y)[0],4)
	print_both("Original average ({0} runs): {1}\nOriginal avg over preds: {2}\nRange: {3} -- {4}\n".format(runs, avg_score_dev, dev_pred_score, min_score_dev, max_score_dev), out_file)
	return avg_preds_dev, avg_preds_test


def translate_train(full_train_X, full_train_Y, dev_X, dev_Y, translated_X, translated_Y, test_X,  arg1, arg2, arg3, arg4, options, options_old, lstm, clf, runs, out_file):
	'''Add translated data to training data and test on dev on average of X runs'''
	## Get training data
	train_trans_X = np.concatenate((full_train_X, translated_X), axis=0)
	train_trans_Y = np.concatenate((full_train_Y, translated_Y), axis=0)
	
	dev_scores = []
	dev_preds = []
	test_preds = []
	start_time = time.time()
	
	## Do training + testing over X runs
	for run in range(runs):
		if lstm:
			dev_pred, dev_score, test_pred = train_lstm(train_trans_X, train_trans_Y, dev_X, dev_Y, test_X, len(train_trans_X[0]), arg1, arg2, arg3, arg4, clf, options, options_old, len(dev_Y))
		else:
			dev_pred, dev_score, test_pred	= train_feedforward(train_trans_X, train_trans_Y, dev_X, dev_Y, test_X, len(train_trans_X[0]), arg1, arg2, arg3, arg4, clf, options, options_old)
		dev_scores.append(dev_score)
		dev_preds.append(dev_pred)
		test_preds.append(test_pred)
	
	## Keep track of results
	dev_avg_score, dev_min_score, dev_max_score = average(dev_scores), min(dev_scores), max(dev_scores)
	dev_avg_preds = np.asarray(dev_preds).mean(axis=0)
	test_avg_preds = np.asarray(test_preds).mean(axis=0)
	dev_avg_pred_score = round(pearsonr(dev_avg_preds, dev_Y)[0],4)
	print_both("Trans average ({0} runs): {1}\nTrans avg over preds: {2}\nRange: {3} -- {4}\n".format(runs, dev_avg_score, dev_avg_pred_score, dev_min_score, dev_max_score), out_file)			
	return dev_avg_preds, test_avg_preds


def silver_train(full_train_X, full_train_Y, dev_X, dev_Y, silver_X, test_X, silver_preds, arg1, arg2, arg3, arg4, clf, options, options_old, lstm, runs, add_sil, out_file):
	''''Train model on both train + silver and average over 10 runs'''
	dev_scores = []
	dev_preds = []
	test_preds = []
	start_time = time.time()
	for run in range(runs):
		## Shuffle silver data, so first add train and test back together, then shuffle, then split again
		silver_preds = np.asarray(silver_preds)
		data = np.append(silver_X, silver_preds.reshape(len(silver_preds), 1), axis=1)
		np.random.shuffle(data)
		silver_X = data[:,0:-1] 
		silver_preds = data[:,-1] 
		
		## Add silver data to gold data and print score
		train_silver_X = np.concatenate((full_train_X, silver_X[0:add_sil]), axis=0)
		train_silver_Y = np.concatenate((full_train_Y, silver_preds[0:add_sil]), axis=0)
		
		## Do feed-forward or LSTM
		if lstm:
			dev_pred, dev_score, test_pred = train_lstm(train_silver_X, train_silver_Y, dev_X, dev_Y, test_X, len(train_silver_X[0]), arg1, arg2, arg3, arg4, clf, options, options_old, len(dev_Y))
		else:
			dev_pred, dev_score, test_pred = train_feedforward(train_silver_X, train_silver_Y, dev_X, dev_Y, test_X, len(train_silver_X[0]), arg1, arg2, arg3, arg4, clf, options, options_old)
		dev_scores.append(dev_score)
		dev_preds.append(dev_pred)
		test_preds.append(test_pred)
	
	## Keep track of results
	dev_avg_score, dev_min_score, dev_max_score = average(dev_scores), min(dev_scores), max(dev_scores)
	dev_avg_preds = np.asarray(dev_preds).mean(axis=0)
	test_avg_preds = np.asarray(test_preds).mean(axis=0)
	dev_avg_pred_score = round(pearsonr(dev_avg_preds, dev_Y)[0],4)
	print_both("Silver average ({0} runs): {1}\nSilver avg over preds: {2}\nRange: {3} -- {4}\n".format(runs, dev_avg_score, dev_avg_pred_score, dev_min_score, dev_max_score), out_file)			
	return dev_avg_preds, test_avg_preds


def pred_to_file(orig_file, predictions, out_dir, ident, emotion, file_type, options, options_old, clf, num_dict):
	'''Write predictions to file in original format so that it works for test'''
	
	out_dir_full = "{0}{1}{2}/".format(out_dir, ident, emotion)
	if not os.path.exists(out_dir_full):
		os.system("mkdir -p {0}".format(out_dir_full))
	
	name = "{0}{1}.txt".format(out_dir_full, file_type)
	
	with open(orig_file, 'r', encoding="utf-8") as infile:
		infile = infile.readlines()
		infile = [x for x in infile if x]
		## If clf we output 3 values: the average regression score, the score it would be normalized to and the string class
		if clf: #+ '\t' + 
			data = ["\t".join(line.split("\t")[:-1]) + "\t" + str(predictions[ix]) + '\t' + str(min(options, key=lambda x:abs(x-predictions[ix]))) + '\t' + num_dict[options_old[options.index(min(options, key=lambda x:abs(x-predictions[ix])))]] for ix, line in enumerate(infile[1:])]
		else:
			data = ["\t".join(line.split("\t")[:-1]) + "\t" + str(predictions[ix]) for ix, line in enumerate(infile[1:])]
		with open(name, 'w', encoding="utf-8") as out:
			out.write(infile[0])
			for line in data:
				out.write(line)
				out.write("\n")
		out.close()


if __name__ == "__main__":
	args = create_arg_parser()
	options = ["anger", "fear", "joy", "sadness", "valence"]
		
	## Get feature vectors from directories
	feature_vectors_train, emotion_order = get_files(args.f1, '.csv', options)
	feature_vectors_dev, _ = get_files(args.d, '.csv', options)
	feature_vectors_test, _ = get_files(args.te, '.csv', options)
	feature_vectors_translated, _ = get_files(args.t, '.csv', options)
	feature_vectors_silver, _ = get_files(args.s, '.csv', options)
	
	## Original dev/test file so we write correct output format
	original_dev, _ = get_files(args.dor, '.txt', options)
	original_test, _ = get_files(args.tor, '.txt', options)
	
	## Get task and check if all files are there
	task = get_task(args.f1, ['EI-Reg','EI-Oc','V-Reg','V-Oc'])
	print (len(feature_vectors_train), len(feature_vectors_dev) , len(feature_vectors_test) , len(feature_vectors_translated))
	assert(len(feature_vectors_train) == len(feature_vectors_dev) ==  len(feature_vectors_test) == len(feature_vectors_translated))
	
	## Set parameters that we know due to parameter searches in feed_forward.py, lstm_search.py and self_training_parameters.py -- note that only EI tasks have parameters related to silver data
	para_dict = {}
	
	if args.lstm: 
		## For LSTM this contains [num_dense_layers, num_lstm_layers, nodes, dropout, similarity_max, add_sil]
		para_dict["EI-Reg-anger"] = [0, 2, 400, 0.001, 0,0]
		para_dict["EI-Reg-fear"] = [1, 2, 400, 0.01, 0,0]
		para_dict["EI-Reg-joy"] = [0, 2, 200, 0.1, 0,0]
		para_dict["EI-Reg-sadness"] = [0, 2, 600, 0.001, 0,0]
		para_dict["EI-Oc-anger"] = [0, 2, 200, 0.001, 0,0]
		para_dict["EI-Oc-fear"] = [0, 2, 200, 0.001, 0,0]
		para_dict["EI-Oc-joy"] = [1, 3, 400, 0.001, 0, 0]
		para_dict["EI-Oc-sadness"] = [1,3,800, 0.01, 0, 0]
		para_dict["V-Reg-valence"] = [1,2, 200, 0.001, 0, 0]
		para_dict["V-Oc-valence"] = [1,3, 600, 0.01, 0, 0]
		
	else:	
		## For feed-forward this contains [dense_layer, last_layer, num_dense_layer, dropout, similarity_max, add_sil]
		para_dict["EI-Reg-anger"] = [600, 200, 0, 0.001, 0.1, 2500]
		para_dict["EI-Reg-fear"] = [700, 200, 0, 0.001, 0.1, 1500]
		para_dict["EI-Reg-joy"] = [500, 500, 0, 0.001, 0.125, 1500]
		para_dict["EI-Reg-sadness"] = [400, 300, 0, 0.001, 0.1, 5000]
		para_dict["EI-Oc-anger"] = [600, 200, 0, 0.001, 0.15, 500]
		para_dict["EI-Oc-fear"] = [700, 300, 0, 0.001, 0.1, 500]
		para_dict["EI-Oc-joy"] = [800, 200, 0, 0.001, 0.125, 500]
		para_dict["EI-Oc-sadness"] = [500, 200, 0, 0.001, 0.125, 500]
		para_dict["V-Reg-valence"] = [400, 400, 1, 0.001, 0, 0]
		para_dict["V-Oc-valence"] = [400, 100, 1, 0.001, 0, 0]
	
	## Other parameters
	
	folds = 10
	runs = 10
	
	out_file = open(args.r, 'w')
	
	if args.lstm:
		alg_type = 'lstm'
	else:
		alg_type = 'feed_forward'
	
	for idx in range(len(feature_vectors_train)):
		print_both("\n\nFor {0}-{1}\n".format(task, emotion_order[idx]), out_file)
		arg1, arg2, arg3, arg4, similarity_max, add_sil = para_dict["{0}-{1}".format(task, emotion_order[idx])]
		
		## Get initial train data (non-shuffled)
		full_train_X, full_train_Y, dataset, options, options_old, num_dict = get_dataset(feature_vectors_train[idx], args.clf, args.shuffle)
		dev_X, dev_Y,_,_,_ , _= get_dataset(feature_vectors_dev[idx], args.clf, args.shuffle)
		test_X, test_Y,_,_,_, _ = get_dataset(feature_vectors_test[idx], False, args.shuffle) #args.clf alwals false for test since we put 0 as category
		translated_X, translated_Y, _, _, _ , _= get_dataset(feature_vectors_translated[idx], args.clf, args.shuffle)
		
		## Get original score (average score over 10 runs + score over averaged predictions over X runs)
		avg_preds_dev, avg_preds_test = get_original_score(full_train_X, full_train_Y, dev_X, dev_Y, test_X,  len(full_train_X[0]), arg1, arg2, arg3, arg4, args.clf, options, options_old, runs, args.lstm, out_file)
		
		## Now add translated data to the training data and test on dev (average of X runs again)
		avg_preds_trans_dev, avg_preds_trans_test  = translate_train(full_train_X, full_train_Y, dev_X, dev_Y, translated_X, translated_Y, test_X, arg1, arg2, arg3, arg4, options, options_old, args.lstm, args.clf, runs, out_file)
		
		if similarity_max != 0 and add_sil != 0:		
			## Filter silver data --> train 10 different models and only keep silver predictions that are similar enough
			silver_X, silver_preds = get_silver_data(feature_vectors_train[idx], feature_vectors_dev[idx], feature_vectors_silver[idx], args.shuffle, runs, similarity_max, arg1, arg2, arg3, arg4,args.clf, out_file, args.lstm)
			
			## Now use that silver data to train new model and test on dev (average of X runs again)
			avg_preds_sil_dev, avg_preds_sil_test = silver_train(full_train_X, full_train_Y, dev_X, dev_Y, silver_X, test_X, silver_preds, arg1, arg2, arg3, arg4, args.clf, options, options_old, args.lstm, runs, add_sil, out_file)
		else:
			print ('Not doing silver because of', similarity_max, add_sil)
		
		## Predictions for training on train and testing on dev + test
		pred_to_file(original_dev[idx], avg_preds_dev, args.o, 'dev/', emotion_order[idx], "{0}_normal".format(alg_type), options, options_old, args.clf, num_dict)
		pred_to_file(original_test[idx], avg_preds_test, args.o, 'test/', emotion_order[idx], "{0}_normal".format(alg_type), options, options_old, args.clf,num_dict)
		
		## Predictions for training on train + translated
		pred_to_file(original_dev[idx], avg_preds_trans_dev, args.o, 'dev/', emotion_order[idx], "{0}_translated".format(alg_type), options, options_old, args.clf, num_dict)
		pred_to_file(original_test[idx], avg_preds_trans_test, args.o, 'test/', emotion_order[idx], "{0}_translated".format(alg_type), options, options_old, args.clf, num_dict)
		
		if similarity_max != 0 and add_sil != 0:
			## Predictions for training on train + silver
			pred_to_file(original_dev[idx], avg_preds_sil_dev, args.o, 'dev/', emotion_order[idx], "{0}_silver".format(alg_type), options, options_old, args.clf, num_dict)
			pred_to_file(original_test[idx], avg_preds_sil_test, args.o, 'test/', emotion_order[idx], "{0}_silver".format(alg_type), options, options_old, args.clf, num_dict)
