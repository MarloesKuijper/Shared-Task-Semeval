#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys,re,os
import json

'''Extract text from tweets dict, format: text TAB username TAB retweet (true/false)
   Run as: python3 twitter_filter.py INFILE OUTFILE'''

if __name__ == "__main__":
	with open(sys.argv[1], "r", encoding="utf-8") as json_data, open(sys.argv[2], "w", encoding="utf-8") as outfile:
		for line in json_data:
			if '"text"' in line:
				tweet = json.loads(line)
				txt = " ".join(tweet["text"].replace('\n',' ').split()) #remove newlines from tweets to not mess ordering up (important) -- also remove double whitespace with split/join
				if txt:
					user = tweet["user"]["screen_name"]
					rt = tweet["retweeted"]
					print_line = (txt.strip() + '\t' + user.strip() + '\t' + str(rt)).strip()
					outfile.write(print_line)
					outfile.write("\n")
					print (print_line)
