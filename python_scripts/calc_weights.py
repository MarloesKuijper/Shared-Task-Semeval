import argparse
import os
from scipy.stats import pearsonr
import numpy as np
import pickle
import itertools
from itertools import permutations
from copy import deepcopy
from collections import defaultdict
import subprocess

'''Script that creates predictions by averaging predictions of models'''

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d","--dev_pred", required=True, type=str, help="Dev predictions folder")
	parser.add_argument("-t","--test_pred", required=True, type=str, help="Test predictions folder")
	parser.add_argument("-g", "--dev_gold", required=True, type=str, help="Dir with gold dev data")
	parser.add_argument("-od", "--orig_dev", required=True, type=str, help="Dir with original test files")
	parser.add_argument("-ot", "--orig_test", required=True, type=str, help="Dir with original test files")
	parser.add_argument("-o","--out_dir", required=True, type=str, help="Dir to write results to")
	parser.add_argument("-ta", "--task_name", choices = ['EI-reg', 'EI-oc', 'V-reg', 'V-oc'], type=str, help="Name of task - choose EI-reg, EI-oc, V-reg or V-oc")
	parser.add_argument("-diff", "--difference", default = 0.002, type=float, help="The amount the F-score should increase by before we discard a model (default 0.1)")
	parser.add_argument("-c", "--clf", action = 'store_true', help="Add this if it is a classification task -- important!")
 
	args = parser.parse_args()
	return args


def calculate_pearson(scores, real_y, weights=None, options=None, old_options=None):
	pred_y = []

	for instance in scores:
		pred_y.append(sum(instance)/len(instance))

	if args.clf:
		scaled_pred_y = []
		for item in pred_y:
			scaled_item = min(options, key=lambda x:abs(x-item))
			scaled_pred_y.append(old_options[options.index(scaled_item)])
		score = round(pearsonr(scaled_pred_y, real_y)[0],4)
		return score

	else:
		score = round(pearsonr(pred_y, real_y)[0],4)

	return score


def remove_if_better(predictions, gold_labels, original_score, model_order, extra_diff, options=None, old_options=None):
	'''Function that tries to remove a model from the averaging and see if it gets better. Do this a lot of times, e.g.
	   first remove the worst model, then try again and see if we can remove one in the current set, etc'''
	print ("Try to remove models from averaging to see if score improves:\n")
	best_diff = 0
	worst_model = -1
	iterations = len(predictions[0])
	for iteration in range(iterations):
		for skip_idx in range(len(predictions[0])):
			new_preds = [x[0:skip_idx] + x[skip_idx + 1:] for x in predictions]
			cur_score = calculate_pearson(new_preds, gold_labels, options=options, old_options = old_options)
			## Check if we score worse, and also if this model is really the worst
			if cur_score > (original_score + extra_diff) and cur_score - original_score > best_diff:
				best_diff = cur_score - original_score
				worst_model = skip_idx
		
		## Set new best score we have to beat
		original_score = original_score + best_diff
		## Throw out worst model from predictions (if we found one), print which one it is      
		if worst_model != -1:
			predictions = [x[0:worst_model] + x[worst_model + 1:] for x in predictions]
			print ("Remove {0} from averaging".format(model_order[worst_model]))
			del model_order[worst_model]
		else:
			print ("\nFound best model with score {0}, includes:".format(original_score))
			for m in model_order:
				print (m)
			print ('\n')
			break #else stop trying, removing a model made it worse 
		
		best_diff = 0
		worst_model = -1
	return original_score, model_order


def cat_to_int(pred):
	'''Convert predicted categories to numbers'''
	new_pred = []
	options = []
	for idx, p in enumerate(pred):
		if p.startswith("'"):
			try:
				new_value = int(p[1])   #predicted category looks something like this: '0: no se infieren niveles de enojo' -- so take second character as number
			except ValueError:
				new_value = int(p[1:3]) #predicted category looks something like this: '-1: no se infieren niveles de enojo' -- so take second + third character as number
		else:
			try:
				# changed this because data is now "'0: no se infieren etc.'"
				new_value = int(p[0])   #predicted category looks something like this: '0: no se infieren niveles de enojo' -- so take second character as number
			except ValueError:
				new_value = int(p[0:2]) #predicted category looks something like this: '-1: no se infieren niveles de enojo' -- so take second + third character as number
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
	return new_Y, sorted(new_options)


def get_file_labels(f, ix):
	'''Get individual labels from file'''
	labels = []
	with open(f, "r", encoding="utf-8") as infile:
		data = infile.readlines()
		for row in data[1:]:
			if args.clf:
				if ix == -1:
					labels.append(row.split("\t")[ix])
				else:
					labels.append(float(row.split("\t")[ix]))
			else:
				labels.append(float(row.split("\t")[ix].strip()))
	return labels           


def fetch_labels(dir, model_types, emotion=None, ix=-1):
	"""" use gold_data_emotion to specify the subtask > necessary to get the right file in the gold_data dir"""
	if emotion:
		files = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".txt") and emotion in f.lower()]
	else:
		files = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".txt")]
	all_labels = []
	model_order = []
	
	## If we get all models
	if model_types:
		for mod in model_types:
			for f in files:
				# make sure order is always the same here and consistent with dir!
				if mod[0] in f:
					model_order.append(mod[1])
					labels = get_file_labels(f, ix)
					all_labels.append(labels)
					
	else:
		labels = get_file_labels(files[0], ix)
		all_labels.append(labels)               
	
	## Getting gold data so only single file
	if emotion:
		return all_labels[0], [], model_order
	## Multiple lists of labels
	else:
		if args.clf and ix == -1:
			options = []
			for labellist in all_labels:
				_, opt = cat_to_int(labellist)
				options.append(opt)
			return list(zip(*all_labels)), options, model_order
		else:
			return list(zip(*all_labels)), [] , model_order


def averaging(all_predictions, indices):
	final_predictions = []
	for instance in all_predictions:
		new_instance = []
		for index, item in enumerate(instance):
			if index in indices:
				new_instance.append(item)
		new_pred = sum(new_instance) / len(new_instance)
		final_predictions.append(new_pred)

	return final_predictions


def reformat_predictions(final_predictions, test_labels, options, old_options):
	'''Reformat the predictions back to categories'''
	rescaled_final_predictions = []
	unique_string_labels = list(set([model for instances in test_labels for model in instances if "infiere" in model]))
	#print("Unique labels:", unique_string_labels)
	for item in final_predictions:
		scaled_item = min(options, key=lambda x:abs(x-item))
		reconverted_score = old_options[options.index(scaled_item)]
		for item in unique_string_labels:
			if item.startswith("'"+str(reconverted_score)):
				# to remove the redundant quotation marks
				item = item.strip()[1:-1]
				rescaled_final_predictions.append(item)
	return rescaled_final_predictions			


def write_output(test_pred, out_dir, emotion, dev_test, original_test_file, task_name):
	'''Write predictions to file'''
	out_dir_full = "{0}{1}/{2}/".format(out_dir,task_name,dev_test)
	if not os.path.exists(out_dir_full):
		subprocess.call(["mkdir", "-p", out_dir_full])
	
	if emotion == "valence":
		name = "{0}{1}_es_pred.txt".format(out_dir_full, emotion)
	else:
		name = "{0}{1}_es_{2}_pred.txt".format(out_dir_full,task_name, emotion)

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


if __name__ == "__main__":
	args = create_arg_parser()
	gold_labels_dir = args.dev_gold
	emotions = ["anger", "fear", "joy", "sadness", "valence"]
	model_types = [["traindev_svm", "SVM Normal"], ["trans_svm","SVM Translated"], ["feed_forward_normal","Feed Forward Normal"], ["feed_forward_translated","Feed Forward Translated"],["feed_forward_silver","Feed Forward Silver"], ["lstm_normal","LSTM normal"], ["lstm_translated","LSTM translated"],["lstm_silver","LSTM silver"]]  
	
	for emotion in emotions:
		## Get correct models for dev/test predictions
		model_dir_dev  = args.dev_pred  + emotion + '/'
		model_dir_test = args.test_pred + emotion + "/"
		assert(model_dir_dev != model_dir_test), "Dev and test folder can not be the same!"
		
		if os.path.exists(model_dir_dev) and os.path.exists(model_dir_test):
			print ("\nDoing tests for {0}:\n".format(emotion))
			if not args.clf:
				## First get predictions + gold labels: order of labels should be consistent
				all_pred_labels, _,  model_order = fetch_labels(model_dir_dev, model_types)
				gold_labels, _, _            	 = fetch_labels(gold_labels_dir, [], emotion)
				all_test_predictions, _, _ 		 = fetch_labels(model_dir_test, model_types)
				
				## Print individual scores of the models to see if some model stands out (very high/low score)
				print ("Individual scores of models\n")
				for idx, model in enumerate(model_order):
					print ('{0}: {1}'.format(model, round(pearsonr([x[idx] for x in all_pred_labels],  gold_labels)[0],4)))

				## Then get the score by averaging
				avg_score = calculate_pearson(all_pred_labels, gold_labels)
				print('\nAveraging all models', avg_score, '\n')
				
				## Keep removing models until it does not help anymore
				best_score_removing, models = remove_if_better(all_pred_labels, gold_labels, avg_score, model_order, args.difference)
				flat_model_types = [model[1] for model in model_types] ## only keep models
				indices_models = [flat_model_types.index(model) for model in models]

				## Write predictions for dev and test
				final_predictions_dev  = averaging(all_pred_labels, indices_models)
				final_predictions_test = averaging(all_test_predictions, indices_models)
				
				## Write output
				original_dev_file  = [os.path.join(args.orig_dev ,f) for f in os.listdir(args.orig_dev)  if os.path.isfile(os.path.join(args.orig_dev, f))  and f.endswith(".txt") and emotion in f.lower()]
				original_test_file = [os.path.join(args.orig_test,f) for f in os.listdir(args.orig_test) if os.path.isfile(os.path.join(args.orig_test, f)) and f.endswith(".txt") and emotion in f.lower()]
				
				write_output(final_predictions_dev,  args.out_dir,  emotion, 'dev' ,original_dev_file[0],  args.task_name)
				write_output(final_predictions_test, args.out_dir,  emotion, 'test',original_test_file[0], args.task_name)
				
			else:
				# Prediction labels for dev
				string_pred_labels, options_pred, model_order = fetch_labels(model_dir_dev, model_types) 		# string pred labels
				scaled_pred_labels, _, _ 					  = fetch_labels(model_dir_dev, model_types, ix=-2) # scaled predictions
				specific_pred_labels, _, _ 					  = fetch_labels(model_dir_dev, model_types, ix=-3) # most specific predictions
				## Gold labels for dev
				gold_labels, _, _ 			= fetch_labels(gold_labels_dir, [], emotion)
				new_gold, old_options 		= cat_to_int(gold_labels)
				scaled_gold_labels, options = rescale(new_gold, old_options)
				## Prediction labels for test
				test_labels, options_test, _ = fetch_labels(model_dir_test, model_types) 	  # string predictions test
				scaled_test_labels, _, _ 	 = fetch_labels(model_dir_test, model_types, ix=-2)   # scaled predictions
				all_test_predictions, _, _   = fetch_labels(model_dir_test, model_types, ix=-3) # most specific predictions

				assert sorted(options_pred) == sorted(options_test), "Options are not the same for pred and test, something is wrong"

				## Print individual scores of the models to see if some model stands out (very high/low score) -- calculate this based on category data (so 4, -2, -1, 0, etc)
				print ("Individual scores of models\n")
				for idx, model in enumerate(model_order):
					cur_labels = [x[idx] for x in string_pred_labels]
					category_int_labels, _ = cat_to_int(cur_labels)
					print ('{0}: {1}'.format(model, round(pearsonr(category_int_labels,  new_gold)[0],4)))
				
				## Print average score
				avg_score = calculate_pearson(specific_pred_labels, scaled_gold_labels, options=options, old_options = old_options)
				print('\nAveraging all models', avg_score, '\n')
				
				## Keep removing models until it does not help anymore
				best_score_removing, models = remove_if_better(specific_pred_labels, scaled_gold_labels, avg_score, model_order, args.difference, options = options, old_options = old_options)
				flat_model_types = [model[1] for model in model_types]
				indices_models = [flat_model_types.index(model) for model in models]
				
				## Get predictions for test and dev
				final_predictions_dev = averaging(specific_pred_labels, indices_models)
				final_predictions_test = averaging(all_test_predictions, indices_models)
				
				## Reformat them back to original categories when writing output
				rescaled_pred_dev = reformat_predictions(final_predictions_dev, test_labels, options, old_options)
				rescaled_pred_test = reformat_predictions(final_predictions_test, test_labels, options, old_options)
				
				## Write output to file
				original_dev_file  = [os.path.join(args.orig_dev ,f) for f in os.listdir(args.orig_dev)  if os.path.isfile(os.path.join(args.orig_dev, f))  and f.endswith(".txt") and emotion in f.lower()]
				original_test_file = [os.path.join(args.orig_test,f) for f in os.listdir(args.orig_test) if os.path.isfile(os.path.join(args.orig_test, f)) and f.endswith(".txt") and emotion in f.lower()]
				
				write_output(rescaled_pred_dev, args.out_dir,  emotion, 'dev', original_dev_file[0],  args.task_name)
				write_output(rescaled_pred_test, args.out_dir, emotion, 'test',original_test_file[0], args.task_name)
