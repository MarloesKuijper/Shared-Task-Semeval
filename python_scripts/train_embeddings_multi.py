# import modules & set up logging
import gensim, logging
from gensim.models.word2vec import LineSentence
import os, re, shlex, sys
import subprocess
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# write reordered embeddings to file
def write_to_file(lst, f):
	with open(f, 'w', encoding="utf-8") as out_f:
		for l in lst:
			out_f.write(l.strip() + "\n")
	out_f.close()	

# reorder the embeddings
def change_columns(f):
	'''Put first column last'''
	new_list = []
	for row in open(f, 'r', encoding="utf-8"):
		spl = row.split() #split on whitespace
		new_list.append("\t".join(spl[1:] + [spl[0]])) #do changing here and join back together with commas to string - commas to actually make it a csv, but idk if it matters
	
	return new_list	
 
# file which contains the preprocessed tweets
# '../embedding_files/1st_batch_prepro.txt'
preprocessed_text_file = sys.argv[1]
# loads data_file as sentences of tokens (automatically splits the data for us)
sentences = LineSentence(preprocessed_text_file)

# ignore for now, this is to merge the first and second batch and train on the total
# sentences2 = LineSentence('../embedding_files/2nd_batch_prepro.txt')
# mergedlist = list(set(sentences + sentences2))


## PARAMETERS TO CREATE
# default of min_count = 5
min_count = [0,5,10]
# default of size = 100
size = [150,200,250]
# default size = ?
window = [1,5,20]
# default = 1
workers = [] # not sure about this
# sg defines the training algorithm > sg=0 is CBOW sg=1 is skipgram. Most seem to use Skip-gram?
sg = [0, 1]
# sample = threshold for configuring which higher-frequency words are randomly downsampled; default is 1e-3, useful range is (0, 1e-5).
sample = []
# alpha is the initial learning rate (will linearly drop to min_alpha as training progresses).
alpha = []

## CREATE FOR LOOP HERE OVER THE DIFFERENT PARAMETERS: e.g. for item in min_count: for item in size etc. and then train model for each combination
for imin in min_count:
	for isize in size:
		for iwin in window:

			# train word2vec on the data > Insert this into your for loop (see above) and change parameters of model here to index of lists above e.g.  min_count=min_count[i] or something
			model = gensim.models.Word2Vec(sentences, min_count=imin, size=isize, window=iwin, workers=1)

			# # save model and change output file name to something that includes the parameter settings e.g. spanish_embeddings_mincount_10_size_100_workers_4.txt
			outfile_name = "../trained_embeddings/1000test"+"_min"+str(imin)+"_size"+str(isize)+"_win"+ str(iwin)+ ".txt"
			model.wv.save_word2vec_format(outfile_name, binary=False)

# this is to load the model: you can play around with it, and check stuff if you suspect something is wrong > e.g. you can print the vocabulary
#new_model = gensim.models.KeyedVectors.load_word2vec_format('tweets_1stbatch.txt')
#vocab = list(new_model.wv.vocab.keys())
#print(vocab)

# directory with the trained embeddings
rootdir = "../trained_embeddings/"
# directory with the reordered embeddings
reordered_dir = "../reordered_embeddings/"

def reorder(rootdir, reordered_dir):
	# go over all files in the dir which houses the trained embeddings, find txt files and reorder
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
			if file.endswith(".txt"):
				fname = os.path.join(subdir, file)
				fname_csv = re.split('[.]', file)[0] + ".csv"
				new_file = change_columns(fname)
				write_to_file(new_file, reordered_dir+fname_csv)
				print("Successfully reordered file")

def gzip_reordered_file(reordered_dir):
	# go over all files in dir with reordered files, gzip them
	for subdir, dirs, files in os.walk(reordered_dir):
		for file in files:
			if file.endswith(".csv"):
				print(file)
				fname = os.path.join(subdir, file)
				command = "gzip " + fname
				args = shlex.split(command)
				print("running gzip")
				p = subprocess.Popen(args) # Success!
				print("gzipping done")

			

reorder(rootdir, reordered_dir)

gzip_reordered_file(reordered_dir)
