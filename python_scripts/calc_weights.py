import argparse
import os
from scipy.stats import pearsonr
import numpy as np
import pickle
from itertools import permutations

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_dir", required=True, type=str, help="Dir with models for specific task (e.g. EI-reg anger")
    parser.add_argument("-test_data_dir", required=True, type=str, help="Dir with test data or dev data")
    parser.add_argument("-subtask", required=True, type=str, help="Name of emotion or valence")

    parser.add_argument("-clf", action = 'store_true', help="Add this if it is a classification task")
 
    args = parser.parse_args()
    return args

def calculate_pearson(scores, real_y, weights=None, options=None):
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

    if args.clf:
        scaled_pred_y = []
        for item in pred_y:
            scaled_item = min(options, key=lambda x:abs(x-item))
            scaled_pred_y.append(scaled_item)
        score = round(pearsonr(scaled_pred_y, real_y)[0],4)
        return score

    else:
        score = round(pearsonr(pred_y, real_y)[0],4)

    return score




def check_strength_scores(scores, real_y, best_score, options=None):
    # checks strength of each model by removing one model each time and checking if score improves or not
    # returns the best score and the best models (index of model numbers starts at 0)
    i=0
    approved_models = []
    models_that_matter = list(range((len(scores[0]))))
    scores = scores
    while i < len(scores[0]):
        model_to_skip = [item for item in models_that_matter if item not in approved_models]
        if model_to_skip:
            model_to_skip = model_to_skip[0]
            new_instances = []
            for instance in scores:
                new_inst = []
                for ix, item in enumerate(instance):
                    if ix != model_to_skip:
                        new_inst.append(item)
                new_instances.append(new_inst)
            new_score = calculate_pearson(new_instances, real_y, options=options)
            if new_score > best_score:
                best_score = new_score
                models_that_matter.remove(model_to_skip)
                scores = new_instances
            else:
                approved_models.append(model_to_skip)
        else:
            # if no more models to skip > means that models_that_matter contains no more items that are irrelevant or have not yet been approved
            return best_score, approved_models
        i = i+1

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



def fetch_labels(dir, gold_data_subtask=None, ix=-1):
    """" use gold_data_emotion to specify the subtask > necessary to get the right file in the gold_data dir"""
    if gold_data_subtask:
        files = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".txt") and gold_data_subtask in f.lower()]
    else:
        files = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".txt")]
    all_labels = []
    print(files)
    for file in files:
        # make sure order is always the same here and consistent with dir!
        labels = []
        with open(file, "r", encoding="utf-8") as infile:
            data = infile.readlines()
            for row in data[1:]:
                if args.clf:
                    if ix == -1:
                        labels.append(row.split("\t")[ix])
                    else:
                        labels.append(float(row.split("\t")[ix]))
                else:
                    labels.append(float(row.split("\t")[ix].strip()))
        all_labels.append(labels)

    if gold_data_subtask:
        return all_labels[0]
    else:
        if args.clf and ix == -1:
            options = []
            for labellist in all_labels:
                #print(labellist)
                _, opt = cat_to_int(labellist)
                options.append(opt)
            return list(zip(*all_labels)), options
        else:
            return list(zip(*all_labels))



if __name__ == "__main__":
    args = create_arg_parser()
    model_dir = args.model_dir
    gold_labels_dir = args.test_data_dir
    if not args.clf:
        # order of labels should be consistent with directory structure!
        all_pred_labels = fetch_labels(model_dir)
        gold_labels = fetch_labels(gold_labels_dir, args.subtask)
        avg_score = calculate_pearson(all_pred_labels, gold_labels)
        print(avg_score)
        best_score, approved_models = check_strength_scores(all_pred_labels, gold_labels, avg_score)
        print(best_score, approved_models)
        #best_score2, best_combo = shuffle_weights(scores, real_y, [0.4, 0.3, 0.2, 0.1])
    else:
        # order of labels should be consistent with directory structure!
        string_pred_labels, options_pred = fetch_labels(model_dir) # string pred labels
        scaled_pred_labels = fetch_labels(model_dir, ix=-2) # scaled predictions
        specific_pred_labels = fetch_labels(model_dir, ix=-3) # most specific predictions
     
        # string gold labels
        gold_labels = fetch_labels(gold_labels_dir, args.subtask)
        new_gold, options_gold = cat_to_int(gold_labels)
        scaled_gold_labels, rescaled_options_gold = rescale(new_gold, options_gold)

        # zijn options gelijk voor elk model en voor pred vs. gold?
        for option in options_pred:
            assert sorted(option) == sorted(options_gold)

        scaled_options = []
        range_divider = len(options_gold) + 1
        for idx, option in enumerate(sorted(options_gold)):
            new_val = round((float(1) / float(range_divider)) * (idx+1), 5)
            scaled_options.append(new_val)

        #print(specific_pred_labels)
        avg_score = calculate_pearson(specific_pred_labels, scaled_gold_labels, options=scaled_options)

        best_score, approved_models = check_strength_scores(specific_pred_labels, scaled_gold_labels, avg_score, options=scaled_options)
        print(best_score, approved_models)
        




    # [[0.5, 0.2, 0.3], [0.55, 0.3, 0.4]]