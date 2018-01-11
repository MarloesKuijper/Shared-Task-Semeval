import os, sys, re, argparse

'''Find keywords used to get the tweets'''

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f1", required=True, type=str, help="File with normal tweets")
	parser.add_argument("-f2", required=True, type=str, help="File with new emotion tweets")
	args = parser.parse_args()
	return args


def get_word_dict(f):
	'''Get dictionary with word counts'''
	word_dict = {}
	for line in open(f,'r'):
		words = line.split()
		for w in words:
			word = w.lower().replace('#','')
			if word in word_dict:
				word_dict[word]+=1
			else:
				word_dict[word] = 1
	return word_dict				

if __name__ == "__main__":
	args = create_arg_parser()
	normal_words = get_word_dict(args.f1)
	scraped_words = get_word_dict(args.f2)
	
	rel_freq = {}
	for s in scraped_words:
		if s in normal_words:
			##Only get keywords if they occured at least 500 times in the tweets with emotion and at least 25 times in tweets that contain all scraped tweets
			if scraped_words[s] > 500 and normal_words[s] > 25:
				freq = float(scraped_words[s]) / float(normal_words[s])
				rel_freq[s] = freq 	#save relative frequency (which words occur relatively more often in emotion set)
	
	
	## Print all words that occur the most often -- relatively speaking
	count = 0
	for w in sorted(rel_freq, key=rel_freq.get, reverse=True):
		if count > 200: #do max 200
			break
		print w#, rel_freq[w], scraped_words[w]
		count += 1
