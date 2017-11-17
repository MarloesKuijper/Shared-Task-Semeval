# coding=utf-8
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
from nltk.corpus import stopwords
import pickle

consumer_key = 'J7pye236tpxMmryUcNZ6fMpcb'
consumer_secret = 'PRx87xz5bPJh59w6qrH4emrWNu5YfjzLD0NSxovLyUbhZi02Za'

access_token = '796625727424753664-DCiUnldLAj9FbCpkFdbHdMoxvpl3XQM'
access_token_secret = 'IwGTPVAwRGyFV6hdhSrfp2IEOij4Eq8kJAlyzgpEmgCkR'

def get_data(json_file):
    with open(json_file) as json_data:
        for line in json_data:
            if 'text' in line:
                tweet = json.loads(line)
                print(tweet["text"])

class Listener(StreamListener):

    def on_data(self, data):
        """append data to json file for that day"""
        try:
            print(data)
            saveFile = open('spanish_tweets.json', 'a')
            saveFile.write(data)
            saveFile.write('\n')
            saveFile.close()
            return True
        except BaseException as e:
            print('failed on data, ', str(e))
            time.sleep(55)

    def on_error(self, status):
        """error handling"""
        print(status)

while True:
    try:
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        twitterStream = Stream(auth, Listener())

        stopwords_es = ["y", "ola", "en", "un", "una", "uno", "unos", "yo", "ya", "haya", "hace", "ha", "estés", "esto", "esta", "el", "ella", "ellos", "ellas", "dos", "donde", "desde", "del", "de", "cuando", "contra", "dentro", "bien", "muy", "cual", "arriba", "aqui", "cierto", "cierta", "como", "con", "al", "a"] 
        twitterStream.filter(languages=["es"], track=stopwords_es)
    except:
        print('Error, just start again')
