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
import math

np.random.seed(2888)

'''Script that implements LSTM neural network for regression'''

## USAGE: python lstm.py -f1 TRAIN_SET -f2 DEV_SET [-scale] [-shuffle]

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


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1", required=True, type=str, help="Train file/folder")
	parser.add_argument("-f2", required=True, type=str, help="Dev file/folder")
	parser.add_argument("-folds", default = 10, type=int, help="Number of folds for cross validation")
	parser.add_argument("-scale", action='store_true', help="If added we scale the features between 0 and 1 (sometimes helps)")
	parser.add_argument("-shuffle", action='store_true', help="If added we shuffle the rows before training (different results for cross validation)")
	parser.add_argument("-clf", action='store_true', help="If added this is a classification task")
	parser.add_argument("-cv", action='store_true', help="If added we do cross validation -- else test on dev set")
	parser.add_argument("-o", required=True, type=str, help="Output file")
	args = parser.parse_args()
	return args


def scale_dataset(dataset):
	'''Scale features of dataset between 0 and 1
	   This apparantly helps sometimes'''
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	return dataset


def load_dataset(f, scale, shuffle):
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	if shuffle: #shuffle for more randomization
		np.random.shuffle(dataset)
	X = dataset[:,0:-1] 
	if scale:
		X = scale_dataset(X)
	Y = dataset[:,-1] 
	return X, Y, dataset  


def cv_dataset(train_X, train_Y,  low, up):
	'''Apparently Keras has no cross validation functionality built in so do it here
	   The rows between low and up are for the validation set, rest is for train set'''
	return_train_X = np.concatenate((train_X[:low, :], train_X[up:, :] ), axis=0)
	return_valid_X = train_X[low:up, :]
	return_train_Y = np.concatenate((train_Y[:low], train_Y[up:] ), axis=0)
	return_valid_Y = train_Y[low:up]
	return return_train_X, return_train_Y, return_valid_X, return_valid_Y


def nearest_value(num, options, old_options):
	'''Get value in list of options which is closest to num'''
	nearest = min(options, key=lambda x:abs(x-num)) 	#nearest value scaled (so 0.25 or 0.5 or something)
	return_value =  old_options[options.index(nearest)]	#nearest value in original numbers	(so 1 or 2 or -2 etc)
	return return_value


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


def get_dataset(f, clf, scale, shuffle):
	'''Get dataset-- use different ways for regression or clf data'''
	if clf:
		train_X, train_Y, dataset, options_train, old_options_train = load_clf_data(f, scale, shuffle)
	else:
		train_X, train_Y, dataset = load_reg_data(f, scale, shuffle)		
	
	## For classification tasks we have to keep track of the number of options
	if clf:
		options = list(set(options_train))
		old_options = list(set(old_options_train))
	else:
		options = []
		old_options = []	
	
	return train_X, train_Y, dataset, sorted(options), sorted(old_options)		


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
	

def load_clf_data(f, shuffle, scale):
	'''Load dataset for classification data'''
	# got very strange errors using pandas so did something else (super ugly but OK)	
	
	## Get dataset and maybe shuffle
	dataset = []
	for line in open(f, 'r'):
		dataset.append(line.split(','))
	if shuffle:
		shuffle(dataset)
	
	## Get X and Y and maybe scale
	X = []
	Y = []
	for line in dataset[1:]:
		X.append(line[0:-1])
		Y.append(line[-1].strip())
	if scale:
		X = scale_dataset(X)
	
	## Convert to int and rescale back
	Y, old_options = cat_to_int(Y)
	Y, options = rescale(Y, old_options)
	return np.asarray(X),np.asarray(Y), np.asarray(dataset), options, old_options


def load_reg_data(f, shuffle, scale):	
	'''Load dataset for regression data'''
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	if shuffle: #shuffle for more randomization
		np.random.shuffle(dataset)
	X = dataset[:,0:-1] 
	if scale:
		X = scale_dataset(X)
	Y = dataset[:,-1]   #select column
	return X,Y, dataset


def train_lstm(train_X, train_Y, dev_X, dev_Y, input_dim, dense_layers, lstm_layers, nodes, dropout, clf, options, old_options):
	# don't use -scale here
	## Reshape data
	train_X = train_X.reshape((len(train_X), 1, len(train_X[0])))
	dev_X = dev_X.reshape((len(dev_X), 1, len(dev_X[0])))
	
	#print (train_X.shape, train_Y.shape, dev_X.shape, dev_Y.shape)
	
	## Get model
	model = Sequential()
	neurons = len(dev_Y)
	
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
	
	## Make predictions and evaluate
	pred = model.predict(dev_X, batch_size=8, verbose=0)
	predictions = [p[0] for p in pred] #put in format we can evaluate, avoid numpy error
	if clf:
		predictions = [nearest_value(y, options, options_old)  for y in predictions] ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
	print('Score: {0}'.format(round(pearsonr(predictions, dev_Y)[0],4)))
	score = round(pearsonr(predictions, dev_Y)[0], 4)
	return predictions, score


def params_lstm(f1, f2, scale, shuffle_bool, folds, output_file, clf, emotion, cv):
	'''Do parameter search for feedforward network'''

	## Set parameters we want to search
	
	#nodes = [50, 100, 150, 200, 400]
	nodes = [200, 400, 600, 800]
	dropouts = [0.001, 0.01, 0.1, 0.2]
	lstm_layers = [2,3]
	dense_layers = [0,1]
	
	## Set some other stuff
	counter = 0
	total_scores = {}
	total_exp = len(nodes) * len(dropouts) * len(lstm_layers) * len(dense_layers)
	
	## Load data	
	train_X, train_Y, dataset, options, options_old = get_dataset(f1, clf, scale, shuffle_bool)
	dev_X, dev_Y, dataset, _, _ = get_dataset(f2, clf, scale, shuffle_bool)
	
	## Get all combinations of parameters
	param_settings = []
	for dense_layer in dense_layers:
		for lstm_layer in lstm_layers:
			for node in nodes:
				for drop in dropouts:
					param_settings.append([dense_layer, lstm_layer, node, drop])
	
	## Save all parameter settings separately and then shuffle so we do it in random order
	## Even if we do not finish we have some more information about works and what does not
	actual_exp = 100
	shuffle(param_settings)
	param_settings = param_settings[0:actual_exp] #only do 100 out of all options
	print ('Perform {0} out of {1} experiments\n'.format(min([actual_exp, total_exp]), total_exp))
	
	## Do parameter search
	for par in param_settings:
		dense_layer, lstm_layer, node, drop = par
		counter += 1
		start_time = time.time()
		
		## Get CV score
		setting = '{0} {1} {2} {3}'. format(dense_layer, lstm_layer , node, drop)
		print (setting,'...')
		
		## If specified we do cross validation, else test on dev set
		if args.cv:
			cv_score = cross_val(train_X, train_Y, dataset, folds, dense_layer, lstm_layer, node, drop, args.clf, options, options_old)
		else:
			_, cv_score = train_lstm(train_X, train_Y, dev_X, dev_Y, len(train_X), dense_layer, lstm_layer, node, drop, clf, options, options_old)
		
		## Keep track of everything
		print ('{2}/{3} -- time {4} -- {0} scores {1}'.format(setting, cv_score, counter, min([actual_exp, total_exp]), round(time.time() - start_time),1))
		total_scores[setting] = cv_score
	
	## Now print ranked results to file 
	with open(output_file, 'a') as out_f:
		out_f.write('\nSorted results for {0}:\n'.format(emotion))
		for w in sorted(total_scores, key=total_scores.get, reverse=True):				
			out_f.write('{0} scores {1}\n'.format(w, total_scores[w]))
	out_f.close()


def is_nan(x):
	'''Check if number is nan https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-in-python'''
	return x != x


def cross_val(train_X, train_Y, dataset, folds, dense_layer, lstm_layer, node, dropout, clf, options, options_old):
	'''Do cross validation for training of neural network'''
	scores = []
	breaker = False
	for fold in range(folds): #do cross-validation
		if not breaker:
			low = int(len(dataset) * fold / folds)
			up  = int(len(dataset) * (fold +1) / folds)
			new_train_X, new_train_Y, new_valid_X, new_valid_Y = cv_dataset(train_X, train_Y, low, up)
			pred, score = train_lstm(new_train_X, new_train_Y, new_valid_X, new_valid_Y, len(train_X[0]), dense_layer, lstm_layer, node, dropout, clf, options, options_old)
			if is_nan(score):
				scores = [0]
				print ("Break search because of nan score")
				breaker = True
				break
			else:
				scores.append(score)
		else:
			scores = [0]		
	
	## Return result
	cv_score = round(float(sum(scores)) / float(len(scores)),4)
	return cv_score


if __name__ == "__main__":
	args = create_arg_parser()
	
	options = ["anger", "fear", "joy", "sadness", "valence"]
		
	## Get files from directories
	feature_vectors_train, emotion_order = get_files(args.f1, '.csv', options)
	feature_vectors_test, emotion_order = get_files(args.f2, '.csv', options)
	
	## Loop over vectors
	for idx,(train_file, test_file) in enumerate(zip(feature_vectors_train, feature_vectors_test)):
		print ('Scores for emotion: {0}'.format(emotion_order[idx]))
		params_lstm(train_file, test_file, args.scale, args.shuffle, args.folds, args.o, args.clf, emotion_order[idx], args.cv)