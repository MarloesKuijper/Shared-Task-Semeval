# -*- coding: utf-8 -*-
import re

f = "../../train_data/Ar-train/txt/2018-EI-reg-Ar-sadness-dev.txt"
o = "../../new_train/ar/2018-EI-reg-ar-sadness-dev.txt"

with open(f, encoding="UTF-8") as infile, open(o, "a+", encoding="UTF-8") as outfile:
	infile = infile.readlines()
	for line in infile:
		line_id, tweet, emotion, intensity = line.split("\t")
		new_tweet = tweet.replace('"', '')
		new_line = "\t".join([line_id, new_tweet, emotion, intensity])
		outfile.write(new_line)

