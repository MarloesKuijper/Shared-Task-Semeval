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
from tqdm import tqdm
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

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--emotion", required=True, type=str, help="Directory with emotion files")
	parser.add_argument("-f","--features", required=True, type=str, help="Directory to save (ALL) extracted feature-files to (or only get features if using --no_extract)")
	parser.add_argument("-best","--bestfeatures", required=True, type=str, help="Directory to save BEST feature-files to (USE DATE AS FOLDER NAME!)")
	parser.add_argument("-emb", "--embeddings", required=True, type=str, help="Embeddings file to use")
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
	    				if emotion in name:
	    					file_list.append(os.path.join(path, name))
	    			else:
	    				file_list.append(os.path.join(path, name))
	    	# if not in lexicon dir (but in emotion dir), if a specific emotion is selected, only take those files related to that emotion
	    	elif emotion:
	    		if emotion in name:
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
				script = './lexicons_test.sh'
			else:
				emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
				#print(emotion_name)
				script = 'lexicons_test.sh'
				# dit zijn windows issues met backslash en forwardslash 
				emotion_file = emotion_file.replace("\\", "/")
				#lexicon = lexicon.replace("\\", "/")
			if "dev" in emotion_file:
				print("extracting for dev set")
				feature_name = feat_dir + lex + "_dev" + emotion_name + ".csv"
			else:
				feature_name = feat_dir + lex + "_" + emotion_name + ".csv"
			# runt een bash script dat de features uit de lexicons haalt
			selection = lexicons_data[lex]
			#print(selection)
			os_call = " ".join([script, emotion_file, feature_name, args.embeddings, selection])
			subprocess.call(os_call, shell=True)
			
	print("Features successfully extracted")



def train_test_pearson(clf, X, Y):
	'''Function that does fitting and pearson correlation with cross-val'''
	res = cross_val_predict(clf, X, Y, cv=10)
	print("Pearson coefficient: {0}\n".format(pearsonr(res,Y)[0]))

	return round(pearsonr(res, Y)[0],4)

def predict_and_write_output(features_train, features_test, original_test_file, outfile):
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
	with open(original_test_file, 'r', encoding="utf-8") as infile:
		infile = infile.readlines()[1:]
		data = [line.rstrip() + "\t" + str(y_guess[ix]) for ix, line in enumerate(infile)]
		with open(outfile, 'w', encoding="utf-8") as out:
			for line in data:
				out.write(line)
				out.write("\n")


def get_svm_results(feature_vectors):
	lex_dict = {}
	
	for f in feature_vectors:
		
		## Get lexicon name
		lexicon_name = f.split("/")[-1].split("_")[0]
		print(lexicon_name)
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
		if lexicon_name in lex_dict:
			lex_dict[lexicon_name].append(pearson_svm)
		else:
			lex_dict[lexicon_name] = [pearson_svm]

	return lex_dict

def get_best_score(result_dict):
	## Get best lexicons (average of 4 scores)	
	best_score = 0	
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
			script = './lexicons_test.sh'
		else:
			emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
			print(emotion_name)
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
			subprocess.call(os_call, shell=True)
		else:
			lexicons = " ".join(all_current_lexicons)
			os_call = " ".join([script, emotion_file, feature_name, args.embeddings, lexicons])
			print(os_call)
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
			script = './lexicons_test.sh'
		else:
			emotion_name = "-".join(re.split("[.]", emotion_file.split("\\")[-1])[0].split("-")[3:])
			print(emotion_name)
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

	return top_lexicons, best_score, feature_names[-1]

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


if __name__ == "__main__":

	args = create_arg_parser()
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
		# you add the best lexicons (that make a difference) here
		lexicons_top = []
		
		# only use for the final 'test' phase to write stuff to extract features of test/dev and write to new file
		#extract_features("../../test_sample/", ["../../test_sample/2018-EI-reg-es-anger-dev.arff", "../../test_sample/2018-EI-reg-es-anger-train.arff"], ["sentistrength"], lexicons)
		#predict_and_write_output("../../test_sample/sentistrength_.csv",  "../../test_sample/sentistrength_dev.csv", "../../test_sample/test/2018-EI-reg-Es-anger-dev.txt", "../../test_sample/predictions.csv")

		## Skip feature extraction if we already did that in a previous run
		if not args.no_extract:	
			extract_features(args.features, emotion_data, lexicons_to_use, lexicons)

		# which lexicon is the best? > starting point
		best_lex, best_score = get_best_starting_lexicon(emotion=emotion_to_test)
		lexicons_top.append(best_lex)
		lexicons_to_use.remove(best_lex)

		# # find optimal set of lexicons
		top_lexicons, top_score, best_feature_vector = find_optimal_lexicon_set(lexicons_to_use, lexicons_top, emotion_data, lexicons, best_score)
		best_feature_vecs.append(best_feature_vector)
		top_lexicons = " ".join(top_lexicons)
		best_lexicons.append(top_lexicons)
		best_scores.append(str(top_score))


	if len(best_feature_vecs) == 4:
		## move these items to feature folder with date of today (args.bestembeddings)
		script = "copy_best_features.sh"
		best_features = " ".join(best_feature_vecs)
		os_call = " ".join([script, args.bestfeatures, best_features])
		subprocess.call(os_call, shell=True)
		# write results to txt file
		with open(args.bestfeatures + "/RESULTS.txt", "w", encoding="utf-8") as outfile:
			for i in range(4):
				text = "Best results emotion {0}: {1} with {2}, feature file {3} with embeddings {4}".format(emotions[i], best_scores[i], best_lexicons[i], best_feature_vecs[i], args.embeddings)
				outfile.write(text)
				outfile.write("\n")



    ## STUFF THAT I'VE TESTED:
    ## test embeddings individually
    # SCORE: 0.588 (anger),  0.528 (fear), 0.617 (joy),  0.5654 (sadness)
    # Test lexicons individually (by running get_best_starting_point and printing all results)
    # SCORE: 
	  #   {'afinn': [0.58230000000000004, 0.53539999999999999, 0.61409999999999998, 0.56789999999999996],
	  # 'bingliu': [0.58740000000000003, 0.53190000000000004, 0.61650000000000005, 0.56689999999999996], 
	  # 'emoticons': [0.58960000000000001, 0.52939999999999998, 0.61699999999999999, 0.56530000000000002], 
	  # 'mpqa': [0.58850000000000002, 0.53080000000000005, 0.61509999999999998, 0.5645],
	  # 'negation': [0.58899999999999997, 0.52900000000000003, 0.61619999999999997, 0.56340000000000001], 
	  # 'nrc10hashsent': [0.59109999999999996, 0.5323, 0.61699999999999999, 0.57310000000000005], 
	  # 'nrc10': [0.58879999999999999, 0.52859999999999996, 0.61760000000000004, 0.5655], 
	  # 'nrchashemo': [0.59130000000000005, 0.56299999999999994, 0.62350000000000005, 0.58540000000000003], 
	  # 's140': [0.59089999999999998, 0.52300000000000002, 0.61519999999999997, 0.57240000000000002], 
	  # 'sentistrength': [0.59360000000000002, 0.53349999999999997, 0.62770000000000004, 0.56369999999999998]}
    ## test embeddings with all lexicons
    # SCORE: 0.5977 (anger),  0.562 (fear), 0.6224 (joy), 0.5684 (sadness)
    # TO DO:
    ## run per emotion
    # anger: best score:  0.6046, best lexicons:  sentistrength mpqa s140 emoticons nrc10hashsent
	# fear: best score:  0.5706, best lexicons:  nrchashemo mpqa bingliu afinn negation sentistrength
	# joy: best score:  0.6337, best lexicons:  sentistrength bingliu afinn emoticons nrchashemo
	# sadness: best score:  0.5862, best lexicons:  nrchashemo bingliu emoticons
	# NRC10 verandert de score helemaal niet > misschien iets mis mee?
