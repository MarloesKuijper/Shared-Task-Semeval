import argparse
import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from scipy.stats import pearsonr
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

'''Script that implements an LSTM network for regression
   Currently doesn't learn much but at least it does something'''

## USAGE: python lstm.py -f1 TRAIN_SET -f2 DEV_SET
def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1", required=True, type=str, help="Train file")
	parser.add_argument("-f2", required=True, type=str, help="Dev file")
	args = parser.parse_args()
	return args


def scale_dataset(dataset):
	'''scale dataset between 0 and 1
	   No idea if this helps but I saved the function if we ever need it'''
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	return dataset


def load_dataset(f):
	dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
	X = dataset[:,0:-1] 
	Y = dataset[:,-1] 
	return X, Y, dataset  


if __name__ == "__main__":
	args = create_arg_parser()
	
	#Load data	
	train_X, train_Y, dataset = load_dataset(args.f1)
	dev_X, dev_Y, _ = load_dataset(args.f2)
	
	## Create model
	input_dim = len(dataset[0])
	model = Sequential()
	model.add(Embedding(input_dim, output_dim=128))
	model.add(LSTM(128))  # return a single vector of dimension
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid')) 						#last layer should output only a single value since we do regression
	model.compile(loss='mse', optimizer='rmsprop', metrics=['mse']) #mse as loss and metric, should be good enough even though it's not pearson

	## Train model
	model.fit(train_X, train_Y, batch_size=16, epochs=10, validation_split = 0.1, verbose=1)
	
	## Make predictions and evaluate
	pred = model.predict(dev_X, batch_size=16, verbose=1)
	predictions = [p[0] for p in pred] #put in format we can evaluate, avoid numpy error
	print('Score: {0}'.format(round(pearsonr(predictions, dev_Y)[0],4)))
