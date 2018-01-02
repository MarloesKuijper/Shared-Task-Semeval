# -*- coding: utf-8 -*-
import re
import sys
import glob, os



def remove_quotes(input, output):
	# write to temp
	with open(input, "r", encoding="UTF-8") as infile, open(output, "w", encoding="UTF-8") as outfile:
		infile = infile.readlines()
		for line in infile:
			line_id, tweet, emotion, intensity = line.split("\t")
			new_tweet = tweet.replace('"', '')
			new_line = "\t".join([line_id, new_tweet, emotion, intensity])
			outfile.write(new_line)
	# write to input file from temp
	with open(output, "r", encoding="utf-8") as infile, open(input, "w", encoding="utf-8") as outfile:
		for line in infile:
			outfile.write(line)


if __name__ == "__main__":
	# find files in certain directory that need to be converted
    file_dir = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]
    remove_quotes(infile, outfile)
   	
