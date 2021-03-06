"""
1. Get folder name with lexicons from sys argv
2. Loop over lexicon files
3. For each lexicon do
4. For each emotion file: RemoveId, Embeddings, Lexicons, Reorder, Save to meaningful name
5. Get all featurevectors and train SVM on them, average per embedding file (for all 4 emotions)
6. Decide which lexicon is best and go on with checking if adding more lexicons increases accuracy or not, if not omit lexicon
7. Currently, the following embeddings are used: trained_emb_mc5_s200_win35_sg0_2018 0.63725 (dec 11)

"""

import os, sys, re, subprocess, shlex, argparse
import numpy as np
import pyexcel as p
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--emotion", required=True, type=str, help="Directory with emotion files")
	parser.add_argument("-f","--features", required=True, type=str, help="Directory to save (ALL) extracted feature-files to (or only get features if using --no_extract)")
	parser.add_argument("-best","--bestfeatures", required=True, type=str, help="Directory to save BEST feature-files to (USE DATE AS FOLDER NAME!)")
	parser.add_argument("-emb", "--embeddings", required=True, type=str, help="Embeddings file to use")
	parser.add_argument("-t", "--test", default = '', type=str, help="If added contains a folder with TEST files that will be processed in same way as training set")
	parser.add_argument("-d", "--dev", default = '', type=str, help="If added contains a folder with DEV files that will be processed in same way as training set")
	parser.add_argument("-tr", "--translated", default = '', type=str, help="If added contains a folder with TRANSLATED files that will be processed in same way as training set")
	parser.add_argument("-c","--clf", action = 'store_true', help="Select this if it is a classification task")
	parser.add_argument("-v","--val", action = 'store_true', help="Select this if it is a valence task")
	parser.add_argument("-n","--no_extract", action = 'store_true', help="We only have to do feature extraction once, by including this parameter we just read the features from --features")
	parser.add_argument("-u","--unix", action = 'store_true', help="If you run on some Linux system there is a different way of splitting paths etc, so then add this")
	args = parser.parse_args()
	return args

def get_files(files, lexicon_dir=False, emotion=""):
	""" returns files from directory (and subdirectories) """ 
	file_list = []
	for path, subdirs, files in os.walk(files):
	    for name in files:
	    	print(name)
	    	if lexicon_dir:
	    		# when loading features from lexicon directory, only take those that have trained on 1 lexicon:
	    		if len(name.split("_")[0].split("-")) == 1:
	    			# if a specific emotion is selected for testing, only take those files pertaining to those emotions
	    			if emotion:
	    				if emotion in name.lower():
	    					file_list.append(os.path.join(path, name))
	    			else:
	    				file_list.append(os.path.join(path, name))
	    	# if not in lexicon dir (but in emotion dir), if a specific emotion is selected, only take those files related to that emotion
	    	elif emotion:
	    		print (emotion, name.lower())
	    		if emotion in name.lower() and name.endswith("arff"):
	    			file_list.append(os.path.join(path, name))
	    	else:
	        	file_list.append(os.path.join(path, name))

	print(file_list)
	return file_list


def extract_features(feat_dir, emotion_data, lexicons_to_use, lexicons_data):
	'''Extract features from files and save in feat_dir'''
	print(emotion_data)
	for lex in lexicons_to_use:
		print(lex)
		for emotion_file in emotion_data:
			## Solve issues with splitting file paths in unix/windows (mac?)
			if args.unix:
				emotion_name = emotion_file.replace('..','').split('.')[0].split('/')[-1]
				if args.clf:
					script = './lexicons_test_classification.sh'
				else:
					script = './lexicons_test.sh'
			else:
				emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
				#emotion_name = "valence"
				#print(emotion_name)
				if args.clf:
					script = 'lexicons_test_classification.sh'
				else:
					script = 'lexicons_test.sh'
				# dit zijn windows issues met backslash en forwardslash 
				emotion_file = emotion_file.replace("\\", "/")
				#lexicon = lexicon.replace("\\", "/")
			if "dev" in emotion_file:
				print("extracting for dev set")
				feature_name = feat_dir + lex + "_dev_" + emotion_name + ".csv"
			else:
				feature_name = feat_dir + lex + "_" + emotion_name + ".csv"
			# runt een bash script dat de features uit de lexicons haalt
			selection = lexicons_data[lex]
			#print(selection)
			os_call = " ".join([script, emotion_file, feature_name, args.embeddings, selection])
			if not os.path.isfile(feature_name): #only do if not exists
				subprocess.call(os_call, shell=True)
			
	print("Features successfully extracted")



def train_test_pearson(clf, X, Y):
	'''Function that does fitting and pearson correlation with cross-val'''
	res = cross_val_predict(clf, X, Y, cv=10)
	print("Pearson coefficient: {0}\n".format(pearsonr(res,Y)[0]))

	return round(pearsonr(res, Y)[0],4)

def train_test_oc(clf, X, Y):
	res = cross_val_predict(clf, X, Y, cv=10)
	return accuracy_score(Y, res)


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
	return np.asarray(new_pred), sorted(options)


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
	
	print ('previous categories', sorted_options)
	print ('current categories', new_options, '\n')
	return new_Y, sorted(new_options)


def get_svm_results(feature_vectors):
	lex_dict = {}
	
	for f in feature_vectors:
		
		## Get lexicon name
		lexicon_name = f.split("/")[-1].split("_")[0]
		print(lexicon_name)
		if not args.clf:
			dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
			X = dataset[:,0:-1] #select everything but last column (label)
			Y = dataset[:,-1]   #select column
			options = []
			old_options = []
		else:
			# dataset = np.genfromtxt(f, delimiter=",", skip_header=1, dtype=None)
			dataset = pd.read_csv(f, skiprows=0)
			X = dataset.iloc[:,0:-1] #select everything but last column (label)
			Y = dataset.iloc[:,-1]   #select column
			Y, old_options = cat_to_int(Y) #change label to int, keep track of different ones
			Y, options = rescale(Y, old_options) #rescale values between 0 and 1 for classification tasks
		# print(X)
		
		print("PREDICTIONS: \n", f)
		## SVM test ##
		from boosting_algorithms import svm_search ##import search function we use here
		epsilon = [0.001, 0.005, 0.35, 0.40] + [float(y * 0.01) for y in range(1,30)] #test very wide range of parameters here
		best_list = svm_search(X, Y, [], [], epsilon, '', args.clf, options, old_options)
		pearson_svm = best_list[0]
		
		#if args.clf:
			## le = LabelEncoder()
			## print("old Y\n", Y)
			## Y = le.fit(Y)
			## print("new Y\n", Y)
			#svm_clf = svm.SVC()
			#print('Training SVM...\n')
			#pearson_svm = train_test_oc(svm_clf, X, Y)
		#else:
			#svm_clf = svm.SVR()
			#print('Training SVM...\n')
			#pearson_svm = train_test_pearson(svm_clf, X, Y)
		
		## Save results in dictionary
		if lexicon_name in lex_dict:
			lex_dict[lexicon_name].append(pearson_svm)
		else:
			lex_dict[lexicon_name] = [pearson_svm]

	return lex_dict

def get_best_score(result_dict):
	## Get best lexicons (average of 4 scores)	
	best_score = -10000	
	for lex in result_dict:
		score = float(sum(result_dict[lex])) / len(result_dict[lex])
		if score > best_score:
			best_score = score
			best_lex = lex

	return best_lex, best_score

def test_all_lexicons_together(lexicons_to_use, emotion_data, lexicons_data, feat_dir):
	feature_vectors = []
	for emotion_file in emotion_data:
		## Solve issues with splitting file paths in unix/windows (mac?)
		if args.unix:
			emotion_name = emotion_file.replace('..','').split('.')[0].split('/')[-1]
			if args.clf:
				script = './lexicons_test_classification.sh'
			else:
				script = './lexicons_test.sh'
		else:
			emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
			print(emotion_name)
			if args.clf:
				script = 'lexicons_test_classification.sh'
			else:
				script = 'lexicons_test.sh'
			# dit zijn windows issues met backslash en forwardslash 
			emotion_file = emotion_file.replace("\\", "/")
		lexicon_names = "-".join(lexicons_to_use)
		feature_name = feat_dir + lexicon_names + "_" + emotion_name + ".csv"
		feature_vectors.append(feature_name)
		# runt een bash script dat de features uit de lexicons haalt
		all_current_lexicons = [lexicons_data[item] for item in lexicons_to_use]
		print(all_current_lexicons)
		print(len(all_current_lexicons), len(lexicons_to_use))
		if "sentistrength" in all_current_lexicons:
			all_current_lexicons.remove("sentistrength")
			lexicons = " ".join(all_current_lexicons)
			print(lexicons)
			os_call = " ".join([script, emotion_file, feature_name, args.embeddings, "sentistrength" + " " + lexicons])
			if not os.path.isfile(feature_name):
				subprocess.call(os_call, shell=True)
		else:
			lexicons = " ".join(all_current_lexicons)
			os_call = " ".join([script, emotion_file, feature_name, args.embeddings, lexicons])
			print(os_call)
			if not os.path.isfile(feature_name):
				subprocess.call(os_call, shell=True)

	lex_dict = get_svm_results(feature_vectors)
	print("lex dict", lex_dict)
	best_lex, best_score = get_best_score(lex_dict)

	return best_score


def check_relevance(top_lexicons, lexicon_to_test, emotion_data, lexicons_data, feat_dir):
	feature_vectors = []
	for emotion_file in emotion_data:
		## Solve issues with splitting file paths in unix/windows (mac?)
		if args.unix:
			emotion_name = emotion_file.replace('..','').split('.')[0].split('/')[-1]
			if args.clf:
				script = './lexicons_test_classification.sh'
			else:
				script = './lexicons_test.sh'
		else:
			emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
			#emotion_name = "valence"
			print(emotion_name)
			if args.clf:
				script = 'lexicons_test_classification.sh'
			else:
				script = 'lexicons_test.sh'
			# dit zijn windows issues met backslash en forwardslash 
			emotion_file = emotion_file.replace("\\", "/")
		lexicon_names = "-".join(top_lexicons) + "-" + lexicon_to_test
		feature_name = feat_dir + lexicon_names + "_" + emotion_name + ".csv"
		feature_vectors.append(feature_name)
		# runt een bash script dat de features uit de lexicons haalt
		selection_top = [lexicons_data[item] for item in top_lexicons]
		selection_new = lexicons_data[lexicon_to_test]
		all_current_lexicons = selection_top
		all_current_lexicons.append(selection_new)
		print(all_current_lexicons)
		
		## Only do if file does not exist
		if not os.path.isfile(feature_name):
			if "sentistrength" in all_current_lexicons:
				all_current_lexicons.remove("sentistrength")
				lexicons = " ".join(all_current_lexicons)
				print(lexicons)
				os_call = " ".join([script, emotion_file, feature_name, args.embeddings, "sentistrength" + " " + lexicons])
				subprocess.call(os_call, shell=True)
			else:
				# selection_top = " ".join([lexicons_data[item] for item in top_lexicons])
				# selection_new = lexicons_data[lexicon_to_test]
				lexicons = " ".join(all_current_lexicons)
				os_call = " ".join([script, emotion_file, feature_name, args.embeddings, lexicons])
				print(os_call)
				subprocess.call(os_call, shell=True)

	lex_dict = get_svm_results(feature_vectors)
	best_lex, best_score = get_best_score(lex_dict)

	return best_score, feature_name

def get_best_starting_lexicon(emotion=""):
	## Get feature vectors from directory for each indiv. lexicon
	feature_vectors = get_files(args.features, lexicon_dir=True, emotion=emotion)
	print(feature_vectors)

	## Run different algorithm on all feature vectors, print results
	result_dict = get_svm_results(feature_vectors)
	print("results\n", result_dict)
	## Get best lexicons (average of 4 scores)	
	best_lex, best_score = get_best_score(result_dict)
	print("Best score: ", best_score, " with: ", best_lex)

	return best_lex, best_score

def find_optimal_lexicon_set(lexicons_to_use, lexicons_top, emotion_data, lexicons, best_score):
	top_lexicons = lexicons_top
	feature_names = []
	for i in range(len(lexicons_to_use)):
		new_score, feature_name = check_relevance(lexicons_top, lexicons_to_use[i], emotion_data, lexicons, args.features)
		#print(new_score)
		if new_score > best_score:
			print("new best score: ", new_score)
			best_score = new_score
			top_lexicons.append(lexicons_to_use[i])
			feature_names.append(feature_name)
			print("new best lexicons: ", " ".join(top_lexicons))
			#lexicons_to_use.remove(lexicons_to_use[i])

	if feature_names:
		return top_lexicons, best_score, feature_names[-1]
	else:
		return top_lexicons, best_score, None

def test_only_embeddings(emotion_data, feat_dir):
	feature_vectors = []
	for emotion_file in emotion_data:
		## Solve issues with splitting file paths in unix/windows (mac?)
		if args.unix:
			emotion_name = emotion_file.replace('..','').split('.')[0].split('/')[-1]
			script = './embedding_test.sh'
		else:
			emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
			print(emotion_name)
			script = 'embedding_test.sh'
			# dit zijn windows issues met backslash en forwardslash 
			emotion_file = emotion_file.replace("\\", "/")
		feature_name = feat_dir + "_onlyemb_" + emotion_name + ".csv"
		feature_vectors.append(feature_name)
		# runt een bash script dat de features uit de lexicons haalt
		os_call = " ".join([script, args.embeddings, emotion_file, feature_name])
		subprocess.call(os_call, shell=True)

	emb_dict = get_svm_results(feature_vectors)
	print("embedding scores", emb_dict)
	best_lex, best_score = get_best_score(emb_dict)

	return best_score


def process_extra_set(best_features, test_folder, name, emotion_to_test, clf, top_lexicons, embeddings):
	'''Process a folder of files in exactly the same way as we did for the training set'''
	found_file = False
	dev_folder = best_features + name
	if not os.path.exists(dev_folder):
		os.makedirs(dev_folder)
	
	for root, dirs, files in os.walk(test_folder):
		for f in files:
			if f.endswith('.arff') and emotion_to_test in f.lower() and not found_file: #check if emotion occurs
				emotion_file = os.path.join(root, f).replace("\\", "/")
				if clf:
					script = './lexicons_test_classification.sh' if args.unix else 'lexicons_test_classification.sh'
				else:
					script = './lexicons_test.sh' if args.unix else 'lexicons_test.sh'
				lexicon_names 	= "-".join(top_lexicons.split())
				feature_name 	= dev_folder + lexicon_names + "_" + emotion_to_test + ".csv"
			
				if not os.path.isfile(feature_name): ## Only do if file does not exist
					if "sentistrength" in top_lexicons:
						add_lexicons = " ".join([lexicons[lex] for lex in top_lexicons.split() if 'sentistrength' not in lex]) #string to add for Weka, dont add sentistrength because we add that anyway
						os_call = " ".join([script, emotion_file, feature_name, embeddings, "sentistrength" + " " + add_lexicons])
						subprocess.call(os_call, shell=True)
					else:
						add_lexicons = " ".join([lexicons[lex] for lex in top_lexicons.split()]) #string to add for Weka
						os_call = " ".join([script, emotion_file, feature_name, embeddings, add_lexicons])
						subprocess.call(os_call, shell=True)
				found_file = True
	if not found_file:
		print ('Specified directory with dev/test files, but could not find a file for emotion {0}'.format(emotion_to_test))


if __name__ == "__main__":

	args = create_arg_parser()
	if args.val:
		emotions = ["valence"]
	else:
		emotions = ["anger", "fear", "joy", "sadness"]
		
	best_feature_vecs = []
	best_lexicons = []
	best_scores = []
	for emotion_to_test in emotions:
		emotion_data = get_files(args.emotion, emotion=emotion_to_test)
		feature_dir = args.features

		lexicons = {"afinn": "-F", "bingliu": "-D", 
				"emoticons": "-R", "mpqa": "-A",
				 "negation": "-T",  "nrc10": "-L",
				 "nrc10expanded": "-N", "nrchashemo": "-P",
				 "nrc10hashsent": "-J", "s140": "-H",
				 "sentiwordnet": "-Q", "sentistrength": "sentistrength"}

		# first you put all lexicons you wanna test in here
		lexicons_to_use = ["mpqa", "bingliu", "afinn", "negation", "s140", "emoticons", "nrc10", "nrc10expanded", "nrchashemo", "nrc10hashsent", "sentistrength", "sentiwordnet"]
		#lexicons_to_use = ["mpqa", "bingliu"]
		# you add the best lexicons (that make a difference) here
		lexicons_top = []
		
		# Skip feature extraction if we already did that in a previous run
		if not args.no_extract:	
			extract_features(args.features, emotion_data, lexicons_to_use, lexicons)

		# which lexicon is the best? > starting point
		best_lex, best_score = get_best_starting_lexicon(emotion=emotion_to_test)
		lexicons_top.append(best_lex)
		lexicons_to_use.remove(best_lex)

		## find optimal set of lexicons
		top_lexicons, top_score, best_feature_vector = find_optimal_lexicon_set(lexicons_to_use, lexicons_top, emotion_data, lexicons, best_score)
		if best_feature_vector:
			best_feature_vecs.append(best_feature_vector)
		else:
			# if only the first lexicon is used
			best_feature_vecs.append(best_lex+"_"+emotion_to_test+".csv")

		top_lexicons = " ".join(top_lexicons)
		best_lexicons.append(top_lexicons)
		best_scores.append(str(top_score))
		
		## Sometimes we also want to process the dev/test files with the lexicon we found for the training set
		## Do this now, since at this point we know the optimal lexicon set for this emotion
		if args.dev:
			process_extra_set(args.bestfeatures, args.dev, 'dev/', emotion_to_test, args.clf, top_lexicons, args.embeddings)
		if args.test:
			process_extra_set(args.bestfeatures, args.test, 'test/', emotion_to_test, args.clf, top_lexicons, args.embeddings)
		
		## Same principle applies to the translated files
		if args.translated:
			process_extra_set(args.bestfeatures, args.translated, 'translated/', emotion_to_test, args.clf, top_lexicons, args.embeddings)	
	
	## Range is different for Val or Reg
	if args.val:
		range_num = 1
	else:
		range_num = 4	
			
	if len(best_feature_vecs) == range_num:
		## move these items to feature folder with date of today (args.bestembeddings)
		print ('Copying best train features to correct location\n')
		##Fix how to call script for unix vs windows
		if args.unix:
			script = './copy_best_features.sh'
		else:
			script = "copy_best_features.sh"
		best_features = " ".join(best_feature_vecs)
		
		## Save train files in train folder
		train_folder = args.bestfeatures + 'train/'
		if not os.path.exists(train_folder):
			os.makedirs(train_folder)
		
		os_call = " ".join([script, train_folder, best_features])
		subprocess.call(os_call, shell=True)
		# write results to txt file
		with open(train_folder + "/RESULTS.txt", "w") as outfile: #removed encoding="utf-8" so it also works in Python2
			for i in range(range_num):
				text = "Best results emotion {0}: {1} with {2}, feature file {3} with embeddings {4}".format(emotions[i], best_scores[i], best_lexicons[i], best_feature_vecs[i], args.embeddings)
				outfile.write(text)
				outfile.write("\n")

