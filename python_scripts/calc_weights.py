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

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred_dev_dir", required=True, type=str, help="Dir with models for specific task, but not emotion (EI-Reg, V-Oc, etc)")
    parser.add_argument("-gold_dev_dir", required=True, type=str, help="Dir with gold dev data")
    parser.add_argument("-test_pred_dir", required=True, type=str, help="Dir with predictions on test")
    parser.add_argument("-orig_test_dir", required=True, type=str, help="Dir with original test files")
    parser.add_argument("-out_dir", required=True, type=str, help="Dir to write results to")
    parser.add_argument("-task_name", required=True, type=str, help="Name of task (e.g. EI-reg")
    parser.add_argument("-clf", action = 'store_true', help="Add this if it is a classification task")
 
    args = parser.parse_args()
    return args

def calculate_pearson(scores, real_y, weights=None, options=None, old_options=None):
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
            scaled_pred_y.append(old_options[options.index(scaled_item)])
        score = round(pearsonr(scaled_pred_y, real_y)[0],4)
        return score

    else:
        score = round(pearsonr(pred_y, real_y)[0],4)

    return score


def check_strength_scores(scores, real_y, best_score, options=None, old_options=None):
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
            new_score = calculate_pearson(new_instances, real_y, options=options, old_options = old_options)
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


def remove_if_better(predictions, gold_labels, original_score, model_order, options=None, old_options=None):
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
            if cur_score > original_score and cur_score - original_score > best_diff:
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


def remove_highest_lowest(predictions, gold_labels, model_order, options=None, old_options=None, final=None):
    '''Function that removes highest and lowest prediction to see if it improves score'''
    new_preds = []
    for preds in predictions:
        cp_preds = list(deepcopy(preds))
        del cp_preds[cp_preds.index(min(cp_preds))] #remove lowest
        del cp_preds[cp_preds.index(max(cp_preds))] #remove highest
        new_preds.append(cp_preds)                  #append new predictions
    if not final:
        high_low_score = calculate_pearson(new_preds, gold_labels, options=options, old_options = old_options)
        print ("Score for removing highest/lowest: {0}\n".format(high_low_score))
        return high_low_score
    else:
        return [sum(instance)/len(instance) for instance in new_preds]
    

def remove_farthest(predictions, gold_labels, model_order, options=None, old_options=None, final=None):
    '''Function that removes furthest prediction to see if it improves score'''
    new_preds = []
    for preds in predictions:
        idx = get_most_far_pred(preds)
        new_preds.append(preds[0:idx] + preds[idx+1:])
    if not final:
        high_low_score = calculate_pearson(new_preds, gold_labels, options=options, old_options = old_options)
        print ("Score for removing farthest away: {0}\n".format(high_low_score))  
        return high_low_score  
    else:
        return [sum(instance)/len(instance) for instance in new_preds]
    

def get_most_far_pred(preds):
    '''Get the prediction that is the most different from all the other predictions'''
    highest_diff = -1
    for idx in range(len(preds)):
        diff = abs(np.mean(preds[0:idx] + preds[idx+1:]) - preds[idx])
        if diff > highest_diff:
            highest_diff = diff
            far_idx = idx
    return far_idx      

    
def shuffle_weights(scores, real_y, weights, options=None, old_options=None):
    # find all combinations of the weights, apply all to the labels, find optimal weight combo
    combinations = list(itertools.permutations(weights, len(weights)))
    best_score = 0
    best_combination = []
    for combo in combinations:
        if args.clf:
            score = calculate_pearson(scores, real_y, combo, options=options, old_options = old_options)
        else:
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

def calculate_final_predictions(all_predictions, best_method, model_types, models=None):
    # model_types needs to be a regular list
    one_step_methods = ["SVM Normal", "SVM Translated", "Feed Forward Normal", "Feed Forward Translated", "Feed Forward Silver",
                        "LSTM Normal", "LSTM Translated", "LSTM Silver", "averaging", "remove_highest_lowest", "remove_farthest"]
    if best_method in one_step_methods:
        # basically if only one model is selected
        if best_method in model_types:
            index = model_types.index(best_method)
            final_predictions = averaging(all_predictions, [index])
        elif best_method == "averaging":
            final_predictions = averaging(all_predictions, models)
        elif best_method == "remove_highest_lowest":
            final_predictions = remove_highest_lowest(all_predictions, gold_labels, model_order, final=True)
        else:
            final_predictions = remove_farthest(all_predictions, gold_labels, model_order, final=True)

    # if method is removing or leave one out we know which models we should include
    else:
        # perform averaging only on those selected models
        final_predictions = averaging(all_predictions, models)

    return final_predictions

def get_best_model_data(emotion_scores, model_types):
            best_score = -1
            best_method = ""
            best_models = []
            for method,method_dict in emotion_scores.items():
                for key,val in emotion_scores[method].items():
                    if key == "score" and val > best_score:
                        best_score = val
                        best_method = method
            if best_method == "leave_one_out" or best_method == "removing":
                best_models = emotion_scores[best_method]["models"]
            else:
                best_models = list(range(0,len(model_types)))

            return best_method, best_models

def write_output(test_pred, out_dir, emotion, original_test_file, task_name):
    
    if args.clf: ##For a classification problem we are not allowed to output 0.45 for example, output nearest value instead
        #test_pred = [nearest_value(y, options, old_options) for y in test_pred] 
        pass
    
    # Write predictions to file
    out_dir_full = "{0}{1}/".format(out_dir, emotion)
    if not os.path.exists(out_dir_full):
        #os.system("mkdir -p {0}".format(out_dir_full))
        subprocess.call(["mkdir", "-p", out_dir_full])
    if emotion == "valence":
        name = "{0}{1}/{2}_es_pred.txt".format(out_dir, emotion, task_name)
    else:
        name = "{0}{1}/{2}_es_{1}_pred.txt".format(out_dir, emotion, task_name)
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
    model_dir = args.pred_dev_dir
    gold_labels_dir = args.gold_dev_dir
    test_pred_dir = args.test_pred_dir
    emotions = ["anger", "fear", "joy", "sadness", "valence"]
    model_types = [["traindev_svm", "SVM Normal"], ["trans_svm","SVM Translated"], ["feed_forward_normal","Feed Forward Normal"], ["feed_forward_translated","Feed Forward Translated"],["feed_forward_silver","Feed Forward Silver"], ["lstm_normal","LSTM normal"], ["lstm_translated","LSTM translated"],["lstm_silver","LSTM silver"]]  
    
    for emotion in emotions:
        emotion_scores = defaultdict(dict)
        model_dir = args.pred_dev_dir + emotion + '/'
        if os.path.exists(model_dir):
            print ("\nDoing tests for {0}:\n".format(emotion))
            if not args.clf:
                ## First get predictions + gold labels: order of labels should be consistent
                all_pred_labels, _,  model_order = fetch_labels(model_dir, model_types)
                gold_labels, _, _            = fetch_labels(gold_labels_dir, [], emotion)
                all_test_predictions, _, _ = fetch_labels(test_pred_dir+emotion+"/", model_types)
                
                ## Print individual scores of the models to see if some model stands out (very high/low score)
                print ("Individual scores of models\n")
                for idx, model in enumerate(model_order):
                    print ('{0}: {1}'.format(model, round(pearsonr([x[idx] for x in all_pred_labels],  gold_labels)[0],4)))
                    emotion_scores[model]["score"] = round(pearsonr([x[idx] for x in all_pred_labels],  gold_labels)[0],4)

                ## Then get the score by averaging
                avg_score = calculate_pearson(all_pred_labels, gold_labels)
                print('\nAveraging all models', avg_score, '\n')
                emotion_scores["averaging"]["score"] = avg_score
                
                
                ## Remove highest and lowest prediction from averaging
                h_l_score = remove_highest_lowest(all_pred_labels, gold_labels, model_order)
                emotion_scores["remove_highest_lowest"]["score"] = h_l_score
                
                ## Remove prediction that is the farthest away from other predictions
                farthest_score = remove_farthest(all_pred_labels, gold_labels, model_order)
                emotion_scores["remove_farthest"]["score"] = farthest_score
                
                ## Then check if it is better to not include all models in the ensemble
                best_score_leave_one_out, approved_models = check_strength_scores(all_pred_labels, gold_labels, avg_score)
                print("Best score (leave-one-out method): {0}\n\nFor models:\n{1}\n".format(best_score_leave_one_out, "\n".join([model_order[x] for x in approved_models])))
                emotion_scores["leave_one_out"]["score"] = best_score_leave_one_out
                emotion_scores["leave_one_out"]["models"] = approved_models
                
                ## Keep removing models until it does not help anymore
                best_score_removing, models = remove_if_better(all_pred_labels, gold_labels, avg_score, model_order)
                # made model_types flat for the next section
                flat_model_types = [model[1] for model in model_types]
                indices_models = [flat_model_types.index(model) for model in models]
                emotion_scores["removing"]["score"] = best_score_removing
                emotion_scores["removing"]["models"] = indices_models
                
                best_method, best_models = get_best_model_data(emotion_scores, flat_model_types)
                print(best_method, best_models)

                # write predictions for test or dev
                write_test = False
                if write_test:
                    final_predictions = calculate_final_predictions(all_test_predictions, best_method, flat_model_types, best_models)
                else:
                    final_predictions = calculate_final_predictions(all_pred_labels, best_method, flat_model_types, best_models)



                original_test_file = [os.path.join(args.orig_test_dir,f) for f in os.listdir(args.orig_test_dir) if os.path.isfile(os.path.join(args.orig_test_dir, f)) and f.endswith(".txt") and emotion in f.lower()]
                

                write_output(final_predictions, args.out_dir, emotion, original_test_file[0], args.task_name)

                
                ## Not doing weights for now
                #best_score2, best_combo = shuffle_weights(all_pred_labels, gold_labels, [0.3, 0.3, 0.2, 0.2])
                #print(best_score2, best_combo)
            else:
                # order of labels should be consistent with directory structure!
                string_pred_labels, options_pred, model_order = fetch_labels(model_dir, model_types) # string pred labels
                scaled_pred_labels, _, _ = fetch_labels(model_dir, model_types, ix=-2) # scaled predictions
                specific_pred_labels, _, _ = fetch_labels(model_dir, model_types, ix=-3) # most specific predictions
                # string gold labels for DEV
                gold_labels, _, _ = fetch_labels(gold_labels_dir, [], emotion)
                new_gold, old_options = cat_to_int(gold_labels)
                scaled_gold_labels, options = rescale(new_gold, old_options)
                # test data
                test_labels, options_test, _ = fetch_labels(test_pred_dir+emotion+"/", model_types) # string predictions test
                scaled_test_labels, _, _ = fetch_labels(test_pred_dir+emotion+"/", model_types, ix=-2) # scaled predictions
                all_test_predictions, _, _ = fetch_labels(test_pred_dir+emotion+"/", model_types, ix=-3) #

                assert sorted(options_pred) == sorted(options_test)

                print(sorted(options_test))

                ## Print individual scores of the models to see if some model stands out (very high/low score) -- calculate this based on category data (so 4, -2, -1, 0, etc)
                print ("Individual scores of models\n")
                for idx, model in enumerate(model_order):
                    cur_labels = [x[idx] for x in string_pred_labels]
                    category_int_labels, _ = cat_to_int(cur_labels)
                    print ('{0}: {1}'.format(model, round(pearsonr(category_int_labels,  new_gold)[0],4)))
                    emotion_scores[model]["score"] = round(pearsonr(category_int_labels,  new_gold)[0],4)
                
                ## Print average score
                avg_score = calculate_pearson(specific_pred_labels, scaled_gold_labels, options=options, old_options = old_options)
                print('\nAveraging all models', avg_score, '\n')
                emotion_scores["averaging"]["score"] = avg_score
                
                ## Remove highest and lowest prediction from averaging
                h_l_score = remove_highest_lowest(specific_pred_labels, scaled_gold_labels, model_order, options=options, old_options=old_options)
                emotion_scores["remove_highest_lowest"]["score"] = h_l_score
                
                ## Remove prediction that is the farthest away from other predictions
                farthest_score = remove_farthest(specific_pred_labels, scaled_gold_labels, model_order, options=options, old_options=old_options)
                emotion_scores["remove_farthest"]["score"] = farthest_score
                
                ## Check if it is better to leave one model out
                best_score_leave_one_out, approved_models = check_strength_scores(specific_pred_labels, scaled_gold_labels, avg_score, options=options, old_options = old_options)
                print("Best score (leave-one-out method): {0}\n\nFor models:\n{1}\n".format(best_score_leave_one_out, "\n".join([model_order[x] for x in approved_models])))
                emotion_scores["leave_one_out"]["score"] = best_score_leave_one_out
                emotion_scores["leave_one_out"]["models"] = approved_models
                
                ## Keep removing models until it does not help anymore
                best_score_removing, models = remove_if_better(specific_pred_labels, scaled_gold_labels, avg_score, model_order, options = options, old_options = old_options)
                flat_model_types = [model[1] for model in model_types]
                indices_models = [flat_model_types.index(model) for model in models]
                emotion_scores["removing"]["score"] = best_score_removing
                emotion_scores["removing"]["models"] = indices_models

                best_method, best_models = get_best_model_data(emotion_scores, flat_model_types)
                print(best_method, best_models)

                # write predictions for test or dev
                write_test = False
                if write_test:
                    final_predictions = calculate_final_predictions(all_test_predictions, best_method, flat_model_types, best_models)
                else:
                    final_predictions = calculate_final_predictions(specific_pred_labels, best_method, flat_model_types, best_models)


                rescaled_final_predictions = []
                unique_string_labels = list(set([model for instances in test_labels for model in instances if "infiere" in model]))
                print(unique_string_labels)
                for item in final_predictions:
                    scaled_item = min(options, key=lambda x:abs(x-item))
                    reconverted_score = old_options[options.index(scaled_item)]
                    for item in unique_string_labels:
                        if item.startswith("'"+str(reconverted_score)):
                            # to remove the redundant quotation marks
                            item = item.strip()[1:-1]
                            rescaled_final_predictions.append(item)


                original_test_file = [os.path.join(args.orig_test_dir,f) for f in os.listdir(args.orig_test_dir) if os.path.isfile(os.path.join(args.orig_test_dir, f)) and f.endswith(".txt") and emotion in f.lower()]
                

                write_output(rescaled_final_predictions, args.out_dir, emotion, original_test_file[0], args.task_name)

                print(args.task_name)

                ## Not doing weights for now    
                #best_score2, best_combo = shuffle_weights(specific_pred_labels, scaled_gold_labels, [0.3, 0.2, 0.0, 0.5], options=scaled_options)
                #print(best_score2, best_combo)

 
                # dont forget
                # spaanse en engelse labels?? 
                # remove if better geeft telkens andere favoriete modellen?
                # wanneer wel en wanneer niet labels die beginnen met een aanhalingsteken? (volgens mij altijd met aanhalingstekens in echte data, alleen zonder in pred files)
                # valence output > nooit een 3 / -3 ?

