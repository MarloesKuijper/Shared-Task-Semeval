"""
1. Get folder name with trained embeddings from sys argv
2. Loop over embedding files
3. For each embedding file do
4. For each emotion file: RemoveId, Embeddings, Reorder, Save to meaningful name
5. Get all featurevectors and train SVM on them, average per embedding file (for all 4 emotions)
6. write results to file

"""

import os, sys, re, subprocess, shlex, argparse
from tqdm import tqdm
import numpy as np
import pyexcel as p
import csv

## Sklearn

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


### USAGE: python test_embeddings.py - w trained_embeddings_dir -e emotion_data_dir  -f feature_dir [--no_extract] [--unix]


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--emotion", required=True, type=str, help="Directory with emotion files")
	parser.add_argument("-w","--word_embeddings", required=True, type=str, help="Directory with word embedding files")
	parser.add_argument("-f","--features", required=True, type=str, help="Directory to save extracted feature-files to (or only get features if using --no_extract)")
	parser.add_argument("-n","--no_extract", action = 'store_true', help="We only have to do feature extraction once, by including this parameter we just read the features from --features")
	parser.add_argument("-u","--unix", action = 'store_true', help="If you run on some Linux system there is a different way of splitting paths etc, so then add this")
	parser.add_argument("-ee","--emb_ext", default = '.csv.gz', type=str, help="Extension of word embedding file (default .csv.gz)")
	args = parser.parse_args()
	return args


def get_files(files, ext):
	""" returns files from directory (and subdirectories) 
		Only get files with certain extension, but if ext == 'all' we get everything """ 
	file_list = []
	for path, subdirs, files in os.walk(files):
		for name in files:
			if name.endswith(ext) or ext == 'all':
				file_list.append(os.path.join(path, name))
	return file_list


def extract_features(feat_dir, trained_embeddings, emotion_data):
	'''Extract features from files and save in feat_dir'''
	
	#Save names with embedding file so that we can find the best embedding set more easily later
	
	for embedding in trained_embeddings:
		for emotion_file in emotion_data:

			## Solve issues with splitting file paths in unix/windows (mac?)
			if args.unix:
				embedding_name = embedding.replace('..','').split('.')[0].split('/')[-1]
				emotion_name = emotion_file.replace('..','').split('.')[0].split('/')[-1]
				script = './embedding_test.sh'
			else:
				embedding_name =  re.split("[.]", embedding.split("\\")[-1])[0]
				emotion_name = re.split("[.]", emotion_file.split("\\")[-1])[0]
				script = 'embedding_test.sh'
				# dit zijn windows issues met backslash en forwardslash 
				emotion_file = emotion_file.replace("\\", "/")
				embedding = embedding.replace("\\", "/")
					
			feature_name = feat_dir + embedding_name + "_" + emotion_name + ".csv"
			
			# runt een bash script dat de features uit de embeddings haalt
			os_call = " ".join([script, embedding, emotion_file, feature_name])
			subprocess.call(os_call, shell=True)
			
	print("Features successfully extracted")


def train_test_pearson(clf, X, Y):
	'''Function that does fitting and pearson correlation
	   Note: added cross validation'''

	res = cross_val_predict(clf, X, Y, cv=10)
	print("Pearson coefficient: {0}\n".format(pearsonr(res,Y)[0]))

	return round(pearsonr(res, Y)[0],4)


if __name__ == "__main__":
	args = create_arg_parser()
	
	## Get files from directories
	trained_embeddings 	= get_files(args.word_embeddings, args.emb_ext)
	emotion_data 		= get_files(args.emotion, '.arff')
	
	## Skip feature extraction if we already did that in a previous run
	if not args.no_extract:	
		extract_features(args.features, trained_embeddings, emotion_data)
	
	## Get feature vectors from directory
	feature_vectors = get_files(args.features, 'all')
	
	## Run different algorithm on all feature vectors, print results
	
	emb_dict = {} #different embeddings
	
	for f in feature_vectors:
		
		## Get embedding type - works for Linux please check if it does for Windows - if it doesn't
		## do a similar if/else construction as in extract_features()
		emb_type = f.split('/')[-1].split('-')[0] 
		dataset = np.loadtxt(f, delimiter=",", skiprows = 1)

		## split into input (X) and output (Y) variables ##
		X = dataset[:,0:-1] #select everything but last column (label)
		Y = dataset[:,-1]   #select column

		print("PREDICTIONS: \n", f)
		## SVM test ##
		svm_clf = svm.SVR()
		print('Training SVM...\n')
		pearson_svm = train_test_pearson(svm_clf, X, Y)
		
		## Save results in dictionary
		if emb_type in emb_dict:
			emb_dict[emb_type].append(pearson_svm)
		else:
			emb_dict[emb_type] = [pearson_svm]
	
	
		
	## Print sorted scores	
	new_dict = {}
	for emb in emb_dict:
		score = float(sum(emb_dict[emb])) / len(emb_dict[emb])
		new_dict[emb] = score
	
	print ('Sorted ranking of scores: \n')
	
	for w in sorted(new_dict, key=new_dict.get, reverse=True):
		print (w, new_dict[w])		
