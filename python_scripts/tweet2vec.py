# NOTE: Script works with Python 2! (Python version 2.7 used by me)

# Script for converting a tweet set to embeddings using word2vec
# Data loading and word2vec procedure are taken from https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html

# Example usage: python tweet2vec.py -i /tweets/spanish/Es-dev.csv -o tweets/spanish/Es-dev_embeddings.csv

#-------------------------------------------------------------------------
# Set up
import argparse
import sys

import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", required = True, type=str, help="Input file")
	parser.add_argument("-o", required = True, type=str, help="Output file")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = create_arg_parser()

	# Loading data
	def ingest():
		data = pd.read_table(args.i, header=None)
		print 'dataset loaded with shape', data.shape    
		return data

	data = ingest()
	data.head(5)

	#-------------------------------------------------------------------------
	#Pre-Processing
	def tokenize(tweet):
		try:
			tweet = unicode(tweet.decode('utf-8').lower())
			tokens = tokenizer.tokenize(tweet)
			return tokens
		except:
			return 'NC'

	def postprocess(data, n=data.shape[0]):
		data = data.head(n)
		data['tokens'] = data[0].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
		data.reset_index(inplace=True)
		data.drop('index', inplace=True, axis=1)
		return data

	data = postprocess(data)

	#-------------------------------------------------------------------------
	# Building word2vec model
	n = data.shape[0] #length of dataset
	x_train, x_test = train_test_split(np.array(data.head(n).tokens),
			                                             test_size=0.2)

	def labelizeTweets(tweets, label_type):
		labelized = []
		for i,v in tqdm(enumerate(tweets)):
			label = '%s_%s'%(label_type,i)
			labelized.append(LabeledSentence(v, [label]))
		return labelized

	x_train = labelizeTweets(x_train, 'TRAIN')
	x_test = labelizeTweets(x_test, 'TEST')

	x_train[2]

	n_dim = 200 #added through helpful comment. !should be argument!
	tweet_w2v = Word2Vec(size=n_dim, min_count=10)
	tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
	tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

	#-------------------------------------------------------------------------
	#storing embeddings
	tweet_w2v.wv.save_word2vec_format(args.o, fvocab=None, binary=False, total_vec=None)

# python tweet2vec.py -i /home/unseen/Documents/University/tweets/Spanish/Scraped/test_filt_tok.txt -o /home/unseen/Documents/University/embeddings/Spanish/test_tok_filt.csv
