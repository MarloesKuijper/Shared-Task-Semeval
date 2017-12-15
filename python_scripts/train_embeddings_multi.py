#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')	#sometimes necessary to avoid unicode errors

# import modules & set up logging
import gensim, logging
from gensim.models.word2vec import LineSentence
import os, re, shlex
import subprocess
import argparse
from collections import Counter

'''Script to train embeddings with multiple parameter settings'''


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", required=True, type=str, help="Directory with txt files for training the embeddings")
	parser.add_argument("-ext", default = '.txt', type=str, help="Extension (file ending maybe more like it) of file in train directory -- only use files with that extension")
	parser.add_argument("-o", required=True, type=str, help="Output directory to save trained embeddings to")
	parser.add_argument("-p", default = 16, type=int, help="How many parallel threads (workers) we use")
	args = parser.parse_args()
	return args


def write_to_file(lst, f):
	with open(f, 'w') as out_f:
		for l in lst:
			out_f.write(l.strip() + "\n")
	out_f.close()	


def change_columns(f):
	'''Put first column last'''
	new_list = []
	for row in open(f, 'r'):
		spl = row.split() #split on whitespace
		if spl:
			new_list.append("\t".join(spl[1:] + [spl[0]])) #do changing here and join back together
		else:
			print ('Empty line in file {0}, skip {1}'.format(f, row))
	return new_list	
 

def reorder(rootdir):
	'''Go over all files in the dir which houses the trained embeddings, find txt files and reorder'''
	for subdir, dirs, files in os.walk(rootdir):
		for f in files:
			if f.endswith(".txt") and 'trained_emb' in f: #check if trained WE file
				fname = os.path.join(subdir, f)
				fname_csv = fname + ".csv"
				if not os.path.isfile(fname_csv):		#only do if not exists
					new_file = change_columns(fname)
					write_to_file(new_file, fname_csv)
	print("Reordered files")


def gzip_reordered_file(reordered_dir):
	'''Go over all files in dir with reordered files, gzip them'''
	for subdir, dirs, files in os.walk(reordered_dir):
		for f in files:
			if f.endswith(".csv") and 'trained_emb' in f:
				fname = os.path.join(subdir, f)
				if not os.path.isfile(fname + '.gz'): #only do if not exists
					command = "gzip -k " + fname
					args = shlex.split(command)
					p = subprocess.Popen(args) 
	print("Gzipping done")


def get_train_data(d, ext):
	'''Get all files we want to keep for training'''
	
	sents = []
	for root, dirs, files in os.walk(d):
		for f in files:
			if f.endswith(ext): #found file to keep
				cur_sents = LineSentence(os.path.join(root, f))
				sents += cur_sents

	#Since we merge files, remove duplicates here
	return list(set(map(tuple, sents)))


def get_stats(tweet_list, output_dir):
	'''Get some simple statistics from data and save them in file'''
	## Create directory if necessary
	subprocess.call("mkdir -p {0}stats".format(output_dir), shell=True)
	
	## Get stats
	num_tweets = len(tweet_list)
	num_tokens = sum([len(x) for x in tweet_list])
	token_dict = Counter(" ".join([" ".join(x) for x in tweet_list]))
	num_unique = len(token_dict)
	num_unique_kept = len([x for x in token_dict if token_dict[x] > 4]) #type has to occur at least 5 times (if we use min_count = 5)
	
	## Print stats to file
	with open("{0}stats/stats.txt".format(output_dir), 'w') as out_f:
		out_f.write("Number of tweets: {0}\n".format(num_tweets))
		out_f.write("Number of tokens: {0}\n".format(num_tokens))
		out_f.write("Number of unique words (case sensitive): {0}\n".format(num_unique))
		out_f.write("Number of unique words kept for min_count = 5: {0}\n".format(num_unique_kept))
	out_f.close()


if __name__ == "__main__":
	args = create_arg_parser()
	
	## Get training data
	print ('Getting and preprocessing all training files...')
	merged_list = get_train_data(args.d, args.ext)
	print ('Doing word2vec with {0} sentences\n'.format(len(merged_list)))
	
	## Get statistics of training data and save to file
	get_stats(merged_list, args.o)
	
	## PARAMETERS TO TEST
	min_count = [5]  			#default 5
	size      = [200]		 	#default 200	
	window    = [25,30,35,40] 	#default 5
	sg        = [0]       		#training algorithm > sg=0 is CBOW sg=1 is skipgram

	## Loop over parameter settings to test all combinations
	for imin in min_count:
		for isize in size:
			for iwin in window:
				for isg in sg:
					##check if model was trained already, if so, skip
					outfile_name = '{0}trained_emb_mc{1}_s{2}_win{3}_sg{4}.txt'.format(args.o, imin, isize, iwin, isg)
					if not os.path.isfile(outfile_name):
						## train word2vec on the data
						model = gensim.models.Word2Vec(merged_list, min_count=imin, size=isize, window=iwin, sg=isg, workers=args.p)

						## save model and change output file name to something that includes the parameter settings 
						model.wv.save_word2vec_format(outfile_name, binary=False)
	
	## Reorder and zip
	reorder(args.o)
	gzip_reordered_file(args.o)
	
	# this is to load a model: you can play around with it, and check stuff if you suspect something is wrong > e.g. you can print the vocabulary
	#new_model = gensim.models.KeyedVectors.load_word2vec_format('tweets_1stbatch.txt')
	#vocab = list(new_model.wv.vocab.keys())
	#print(vocab)
