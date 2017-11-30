"""
1. Get folder name with trained embeddings from sys argv
2. Loop over embedding files
3. For each embedding file do
4. For each emotion file: RemoveId, Embeddings, Reorder, Save to meaningful name
5. Get all featurevectors and train SVM on them, average per embedding file (for all 4 emotions)
6. write results to file

"""

import os, sys, re, subprocess, shlex
## Keras

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

## Sklearn

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tqdm import tqdm
import numpy as np
import pyexcel as p
import csv

### USAGE: python test_embeddings.py trained_embeddings_dir emotion_data_dir feature_dir

# zonder trailing /
trained_embeddings_dir = sys.argv[1]
# met es van espanol
emotion_data_dir = sys.argv[2]
# met trailing /
feature_dir = sys.argv[3]

def get_files(files):
	""" returns files from directory (and subdirectories) """ 
	file_list = []
	for path, subdirs, files in os.walk(files):
	    for name in files:
	        file_list.append(os.path.join(path, name))
	return file_list

trained_embeddings = get_files(trained_embeddings_dir)
emotion_data = get_files(emotion_data_dir)

def extract_features(feat_dir, trained_embeddings, emotion_data):
	for embedding in trained_embeddings[:1]:
		for emotion_file in emotion_data[:1]:
			print(emotion_file)
			embedding_name =  re.split("[.]", embedding.split("\\")[-1])[0]
			emotion_name = re.split("[.]", emotion_file.split("\\")[-1])[0]
			feature_name = feat_dir + embedding_name + "_" + emotion_name + ".csv"

			# dit zijn windows issues met backslash en forwardslash 
			emotion_file = emotion_file.replace("\\", "/")
			embedding = embedding.replace("\\", "/")

			# runt een bash script dat de features uit de embeddings haalt
			subprocess.call(["embedding_test.sh", embedding, emotion_file, feature_name], shell=True)
			
			print("features successfully extracted")



#extract_features(feature_dir, trained_embeddings, emotion_data)

def train_test_pearson(clf, X_train, y_train, X_test, y_test):
	'''Function that does fitting and pearson correlation'''
	clf.fit(X_train, y_train)
	res = clf.predict(X_test)
	print("Pearson coefficient: {0}\n".format(pearsonr(res,y_test)[0]))

	return pearsonr(res, y_test)[0]

def baseline_model(nodes, input_dim):
	# create model
	model = Sequential()
	model.add(Dense(nodes, input_dim = input_dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



feature_vectors = get_files(feature_dir)

for file in feature_vectors:
	dataset = np.loadtxt(file, delimiter=",", skiprows = 1)

	## split into input (X) and output (Y) variables ##
	X = dataset[:,0:-1] #select everything but last column (label)
	Y = dataset[:,-1]   #select column
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

	print("PREDICTIONS ", file)
	## SVM test ##
	svm_clf = svm.SVR()
	print('Training SVM...\n')
	pearson_svm = train_test_pearson(svm_clf, X_train, y_train, X_test, y_test)

	## Running baseline neural model ##
	print('Training neural baseline...\n')
	input_dim = len(X_train[0]) #input dimension is a necessary argument for the baseline model
	estimator = KerasRegressor(build_fn=baseline_model, nodes = 150, input_dim = input_dim, nb_epoch=100, batch_size=5, verbose=0)
	pearson_neural = train_test_pearson(estimator, X_train, y_train, X_test, y_test)
