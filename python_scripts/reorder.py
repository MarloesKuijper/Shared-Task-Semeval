# -*- coding: utf8 -*-

'''
Put first column last for each line in input file
'''


import argparse
import sys, os, re



def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", required=True, type=str, help="Input file")
	parser.add_argument("-o", required=True, type=str, help="Outputfile")
	args = parser.parse_args()
	return args


def write_to_file(lst, f):
	with open(f, 'w', encoding="utf-8") as out_f:
		for l in lst:
			out_f.write(l.strip() + "\n")
	out_f.close()	


def change_columns(f):
	'''Put first column last'''
	new_list = []
	for row in open(args.f, 'r', encoding="utf-8"):
		spl = row.split() #split on whitespace
		new_list.append("\t".join(spl[1:] + [spl[0]])) #do changing here and join back together with commas to string - commas to actually make it a csv, but idk if it matters
	
	return new_list	

	
if __name__ == "__main__":
	args = create_arg_parser()
	print("args returned")
	new_list = change_columns(args.f)
	print("colums changed")
	write_to_file(new_list, args.o) #write output	
	
