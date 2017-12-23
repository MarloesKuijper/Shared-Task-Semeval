import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import os
import numpy as np

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--features", required=True, type=str, help="Directory with all files with extracted features (vectors)")
	parser.add_argument("-e","--extension", default = '.csv', type=str, help="Extension of feature vector files (default .csv)")
	args = parser.parse_args()
	return args

def train_test_pearson(X, Y, clf):
	'''Function that does fitting and pearson correlation
	   Note: added cross validation'''

	res = cross_val_predict(clf, X, Y, cv=10)
	return round(pearsonr(res, Y)[0],4)

def get_files(files, ext):
	""" returns files from directory (and subdirectories) 
		Only get files with certain extension""" 
	file_list = []
	for path, subdirs, files in os.walk(files):
		for name in files:
			if name.endswith(ext):
				file_list.append(os.path.join(path, name))
	return file_list


def baseline_model(nodes, input_dim):
	# create model
	model = Sequential()
	model.add(Dense(nodes, input_dim = input_dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


if __name__ == "__main__":
	args = create_arg_parser()
	
	## Get files from directories
	feature_vectors = get_files(args.features, args.extension)
	
	##Loop over different scores per file
	for f in feature_vectors:
		dataset = np.loadtxt(f, delimiter=",", skiprows = 1)

		## split into input (X) and output (Y) variables ##
		X = dataset[:,0:-1] #select everything but last column (label)
		Y = dataset[:,-1]   #select column
		
		print('Training neural baseline...\n')
		input_dim = len(X[0]) #input dimension is a necessary argument for the baseline model
		estimator = KerasRegressor(build_fn=baseline_model, nodes = 150, input_dim = input_dim, nb_epoch=100, batch_size=5, verbose=1)
		score = train_test_pearson(X, Y, estimator)
		print(score)