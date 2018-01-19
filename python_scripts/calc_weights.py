import argparse
import os
from scipy.stats import pearsonr
import numpy as np
import pickle
from itertools import permutations

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-scores", required=True, type=str, help="Pickle file with the scores of all models as a list of lists")
    parser.add_argument("-real_y", required=True, type=str, help="Pickle file with the real labels as list")
    parser.add_argument("-clf", action = 'store_true', help="Add this if it is a classification task")
 
    args = parser.parse_args()
    return args

def calculate_pearson(scores, real_y, weights=None):
	pred_y = []
	# [0.3, 0.4, 0.3]
	if weights:
		for instance in scores:
			instance_scores = []
			for ix, item in enumerate(instance):
				instance_scores.append(item * weights[ix])
			pred_y.append(sum(instance_scores))
	else:
		# take average
		for instance in scores:
			pred_y.append(sum(instance)/len(instance))

	score = round(pearsonr(pred_y, real_y)[0],4)

	return score




def check_strength_scores(scores, real_y, best_score):
	# checks strength of each model by removing one model each time and checking if score improves or not
	# returns the best score and the best models (index of model numbers starts at 0)
	i=0
	approved_models = []
	models_that_matter = range((len(scores[0])))
	scores = scores
	while i < len(scores[0]):
		model_to_skip = [item for item in models_that_matter if item not in approved_models]
		if model_to_skip:
			model_to_skip = model_to_skip[0]
			new_instances = []
			for instance in scores:
				new_inst = []
				for ix, item in enumerate(instance):
					if ix not in model_to_skip:
						new_inst.append(item)
				new_instances.append(new_inst)
			new_score = calculate_pearson(new_instances, real_y)
			if new_score > best_score:
				best_score = new_score
				models_that_matter.remove(model_to_skip)
				scores = new_instances
			else:
				approved_models.append(model_to_skip)
		else:
			# if no more models to skip > means that models_that_matter contains no more items that are irrelevant or have not yet been approved
			return best_score, approved_models
		i++

	return best_score, approved_models


def shuffle_weights(scores, real_y, weights):
	# find all combinations of the weights, apply all to the labels, find optimal weight combo
	combinations = list(itertools.permutations(weights, len(weights)))
	best_score = 0
	best_combination = []
	for combo in combinations:
		score = calculate_pearson(scores, real_y, combo)
		if score > best_score:
			best_score = score
			best_combination = combo

	return best_score, best_combination


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


def reconvert_y(Y, options_original, options_original_txt):
	sorted_options = sorted(set(Y))
	range_divider = len(options_original) + 1
	new_options = []

	for idx, option in enumerate(sorted_options):
		new_val = round((option * float(range_divider)) * (idx+1), 5)
		new_options.append(new_val)

	new_Y = []
	for y in Y:
		new_Y.append(new_options[sorted_options.index(y)])

	new_pred = []
	for y in new_Y:
		for option in options_original_txt:
			if option.startswith(y):
				new_pred.append(option)

	assert len(new_pred) == len(new_Y)

	return new_pred




if __name__ == "__main__":
    args = create_arg_parser()
    scores = pickle.load(args.scores)
    real_y = pickle.load(args.real_y)
    if clf:
    	real_y = reconvert_y(real_y, options_original, options_original_txt)
    avg_score = calculate_pearson(scores, real_y)
    best_score, approved_models = check_strength_scores(scores, real_y, avg_score)

    best_score2, best_combo = shuffle_weights(scores, real_y, [0.4, 0.3, 0.2, 0.1])



    ## TO DO: 
    # incorporate into the prediction files: write scores to pickle
    # write functions to fetch original options for certain (OC) task from original .txt or .arff



    # [[0.5, 0.2, 0.3], [0.55, 0.3, 0.4]]