import re, os, sys
from nltk.tokenize import TweetTokenizer

def url_to_placeholder(tweet):
    tweet= re.sub(r"http\S+", "url", str(tweet))
    return tweet

def number_to_placeholder(tweet):
    tweet = re.sub(r'[0-9]+', "num", str(tweet))
    return tweet

def reference_to_placeholder(tweet):
    tweet = re.sub(r'@([A-Za-z0-9_]+)', "@username", str(tweet))
    return tweet

def min_length(tweet):
    tweet_tokens = tweet.split()
    if len(tweet_tokens) >= 10:
        return True

def remove_duplicates(tweets):
    return list(set(tweets))

def retweet_remove(tweet):
    tweet = re.sub(r'RT @username[:]?', '', str(tweet))
    return tweet

def tokenize_tweet(tweet):
    tokenizer = TweetTokenizer(reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    return " ".join(tokens)

def clean_tweets(tweets):
    tweets_final = []
    for line in tweets:
        # fetch only the tweet, not metadata
        #tweet = line.split("\t")[0]
        tweet = tokenize_tweet(line)
        cleaned_tweet = retweet_remove(url_to_placeholder(number_to_placeholder(reference_to_placeholder(tweet))))
        if min_length(cleaned_tweet):
            tweets_final.append(cleaned_tweet)

    return tweets_final

def find_arffs_in_dir(directory):
    return [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith(".arff")]

def preprocessing_for_arff_files(directory):
    files = find_arffs_in_dir(directory)
    print(files)
    for file in files:
        lines = []
        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                if line != "\n" and not line.startswith("@"):
                    items = line.split(",")
                    tweet = items[1]
                    tokenized = tokenize_tweet(tweet)
                    cleaned = retweet_remove(url_to_placeholder(number_to_placeholder(reference_to_placeholder(tokenized))))
                    items[1] = cleaned
                    new_items = ",".join(items)
                    lines.append(new_items)
                else:
                    lines.append(line)
        with open(file, "w", encoding="utf-8") as outfile:
            for line in lines:
                outfile.write(line)


                    



""" NORMAL USAGE: python3 tweet_preprocessing.py INFILE OUTFILE
 If you want to preprocess arff files (not necessary anymore since it's already done), you can call preprocessing_for_arff_files with a directory of arffs as input"""

if __name__ == "__main__":
    preprocessing_for_arff = False
    if preprocessing_for_arff:
        preprocessing_for_arff_files("./files_to_convert")
    else:
        with open(sys.argv[1], "r", encoding="utf-8") as filtered_tweets, open(sys.argv[2], "w", encoding="utf-8") as outfile:
            tweets_final = clean_tweets(filtered_tweets)
            tweets_final = remove_duplicates(tweets_final)
            for item in tweets_final:
                outfile.write(item.strip())
                outfile.write("\n")
    
    