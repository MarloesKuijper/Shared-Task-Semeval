import os, sys, re, subprocess, shlex, argparse
import numpy as np
from sklearn import svm
from scipy.stats import pearsonr
from random import uniform

'''Loop over all SemEval scraped tweets and save them to set of silver joy/anger/fear/sadness if they contain one of the keywords for that emotion'''

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-a", required=True, type=str, help="File with annotated words")
	parser.add_argument("-t", required=True, type=str, help="File with all scraped tweets from SemEval")
	parser.add_argument("-f", required=True, type=str, help="Folder to save new files to")
	args = parser.parse_args()
	return args


def create_emotion_dict(annotated_words, emo):
	'''Create dictionary that contains emotion so we can search faster'''
	d = {}
	for a in annotated_words:
		if a[1] == emo:
			d[a[0]] = 1 #save word in dictionary
	return d		

def contains_any(tweet, d):
	'''Check if tweet contains at least one the words'''
	for t in tweet:
		if t in d:
			return True
	return False		

if __name__ == "__main__":
	args = create_arg_parser()
	
	emotions = ['sadness', 'joy', 'fear', 'anger']
	
	annotated_words = [x.split() for x in open(args.a,'r')]
	tweets = [x.strip() for x in open(args.t,'r')]
	
	## Get specific sets
	sadness = create_emotion_dict(annotated_words, 's')
	joy 	= create_emotion_dict(annotated_words, 'j')
	fear 	= create_emotion_dict(annotated_words, 'f')
	anger 	= create_emotion_dict(annotated_words, 'a')
	d_col   = [sadness, joy, fear, anger] #put dictionary in a list so we can loop over them
	d_save  = [[], [], [], []] 
	
	## Loop over tweets and put each tweet in correct emotion set (if possible)
	for idx, tweet in enumerate(tweets):
		tweet_check = tweet.replace('#',' # ').split()
		res_list = []
		for d in d_col:
			## Check if the tweet contains one of the words in the current dictionary (sad, joy, fear, anger) and save results
			if contains_any(tweet_check, d):
				res_list.append(1)
			else:
				res_list.append(0)
		
		if sum(res_list) == 0:  #no words appear, don't save
			continue
		elif sum(res_list) > 1: #multiple words appear for conflicting emotions, do not save	
			continue
		else:                   ##Save tweet in correct dictionary if not exists yet	
			d_save[res_list.index(1)].append(tweet)
	
	## Save silver emotion tweets to file
	for idx, em in enumerate(emotions):
		with open(args.f + 'silver_{0}.txt'.format(em),'w') as out_f:
			for tweet in d_save[idx]:
				out_f.write(tweet.strip() + '\n')
		out_f.close()				
