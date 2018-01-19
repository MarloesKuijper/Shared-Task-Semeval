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





if __name__ == "__main__":
    args = create_arg_parser()
    scores = pickle.load(args.scores)
    real_y = pickle.load(args.real_y)
    avg_score = calculate_pearson(scores, real_y)
    best_score, approved_models = check_strength_scores(scores, real_y, avg_score)

    best_score2, best_combo = shuffle_weights(scores, real_y, [0.4, 0.3, 0.2, 0.1])





    # [[0.5, 0.2, 0.3], [0.55, 0.3, 0.4]]