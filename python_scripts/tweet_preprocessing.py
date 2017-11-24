import re
import sys

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

def clean_tweets(tweets):
    tweets_final = []
    for line in tweets:
        tweet = line.split("\t")[0]
        cleaned_tweet = retweet_remove(url_to_placeholder(number_to_placeholder(reference_to_placeholder(tweet))))
        if min_length(cleaned_tweet):
            tweets_final.append(cleaned_tweet)

    return tweets_final

""" USAGE: python3 pythonfile INFILE OUTFILE"""
if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding="utf-8") as filtered_tweets, open(sys.argv[2], "w", encoding="utf-8") as outfile:
        tweets_final = clean_tweets(filtered_tweets)
        tweets_final = remove_duplicates(tweets_final)
        for item in tweets_final:
            outfile.write(item.strip())
            outfile.write("\n")