import numpy as np 
from sklearn import svm 
import argparse
import os

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1","--feat_train", required=True, type=str, help="Features dir training data")
	parser.add_argument("-f2","--feat_test", required=True, type=str, help="Features dir test data")
	parser.add_argument("-to","--original_test", required=True, type=str, help="Dir with original .txt test files")
	parser.add_argument("-d","--out_dir", required=True, type=str, help="Directory where you want the results to be written to")
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

def predict_and_write_output(features_train, features_test, original_test_file, out_dir):
	""" takes feature vector for train, feature vector for test and outfile name, 
	trains svm training data, it predicts the labels for the test data and writes this to a new file in the format of the 
	original Xtest_file (same as Xtest_file but with 1 extra column)"""
	
	## Get lexicon name
	lexicon_name = features_train.split("/")[-1].split("_")[0]
	print(lexicon_name)
	train_dataset = np.loadtxt(features_train, delimiter=",", skiprows = 1)

	## split into input (X) and output (Y) variables ##
	Xtrain = train_dataset[:,0:-1] #select everything but last column (label)
	Ytrain = train_dataset[:,-1]   #select column

	test_dataset = np.loadtxt(features_test, delimiter=",", skiprows=1)
	# for now we take off the label, but with the real test data we don't have the label
	Xtest = test_dataset[:, 0:-1] 
	
	## SVM test ##
	clf = svm.SVR()
	y_guess = clf.fit(Xtrain, Ytrain).predict(Xtest)

	options = ["anger", "fear", "joy", "sadness", "valence"]
	name = out_dir + "/" + " ".join([item for item in options if item in features_train]) + "-pred.txt"
	with open(original_test_file, 'r', encoding="utf-8") as infile:
		infile = infile.readlines()[1:]
		data = [line.rstrip() + "\t" + str(y_guess[ix]) for ix, line in enumerate(infile)]
		with open(name, 'w', encoding="utf-8") as out:
			for line in data:
				out.write(line)
				out.write("\n")


if __name__ == "__main__":
	args = create_arg_parser()
	train_dir = args.feat_train
	test_dir = args.feat_test
	original_test = args.original_test
	out_dir = args.out_dir

	training_feats = get_files(train_dir, ".csv")
	test_feats = get_files(test_dir, ".csv")
	original_txts = get_files(original_test, ".txt")
	original_txts = [original_txts[2], original_txts[3], original_txts[1], original_txts[0]]

	print(training_feats)
	print(test_feats)
	print(original_txts)

	for ix, file in enumerate(training_feats):
		predict_and_write_output(training_feats[ix], test_feats[ix], original_txts[ix], out_dir)

