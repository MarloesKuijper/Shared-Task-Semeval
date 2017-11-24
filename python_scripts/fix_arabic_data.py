import argparse

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", required = True, type=str, help="Input folder")
	parser.add_argument("-o", required = True, type=str, help="Output folder")
	args = parser.parse_args()
	return args



def bingliu_to_arff(infile, outfile, delimiter="\t", parts=4):
	"""first remove all instructions at the top, only provide the lines of actual data of the file """
	with open(infile, "r", encoding="utf-8") as before, open(outfile, "w", encoding="utf-8") as after:
		header='@relation '+infile+'\n\n@attribute english string \n@attribute arabic string\n@attribute buckwalter string\n@attribute sentscore numeric \n\n@data\n'
		after.write(header)
		for line in before:
			print(line)
			parts = line.split(delimiter)
			english = parts[0]
			arabic = parts[1]
			buckwalter = parts[2]
			sentiment = parts[3]
			new_line = '"{0}","{1}","{2}",{3}'.format(english, arabic, buckwalter, sentiment)
			after.write(new_line)

def nrc_to_arff(infile, outfile, delimiter="", parts=5):
	"""first remove all instructions at the top, only provide the lines of actual data of the file """
	with open(infile, "r", encoding="utf-8") as before, open(outfile, "w", encoding="utf-8") as after:
		header='@relation '+infile+'\n\n@attribute english string \n@attribute emotion string\n@attribute score numeric\n@attribute arabic string\n@attribute bw string \n\n@data\n'
		after.write(header)
		for line in before:
			print(line)
			parts = line.split()
			english = parts[0]
			emotion = parts[1]
			score = parts[2]
			arabic = parts[3]
			bw = parts[4]
			new_line = '"{0}","{1}",{2},"{3}","{4}"'.format(english, emotion, score, arabic, bw)
			after.write(new_line)
			after.write("\n")

def nrc_hashtag_s140_to_arff(infile, outfile, delimiter="", parts=5):
	"""first remove all instructions at the top, only provide the lines of actual data of the file """
	with open(infile, "r", encoding="utf-8") as before, open(outfile, "w", encoding="utf-8") as after:
		header='@relation '+infile+'\n\n@attribute english string \n@attribute arabic string\n@attribute sent_score numeric\n@attribute pos_occ numeric\n@attribute neg_occ numeric \n\n@data\n'
		after.write(header)
		for line in before:
			print(line)
			parts = line.split("\t")
			english = parts[0]
			arabic = parts[1]
			sent_score = parts[2]
			pos_occ = parts[3]
			neg_occ = parts[4]
			new_line = '"{0}","{1}",{2},{3},{4}'.format(english, arabic, sent_score, pos_occ, neg_occ)
			after.write(new_line)
			after.write("\n")


def mpqa_to_arff(infile, outfile, delimiter=" ", parts=8):
	with open(infile, "r", encoding="utf-8") as before, open(outfile, "w", encoding="utf-8") as after:
		header='@relation '+infile+'\n\n@attribute type string \n@attribute len string\n@attribute word1 string\n@attribute pos1 string\n@attribute stemmed string\n@attribute priorpolar string\n@attribute arabic string\n@attribute bw string \n\n@data\n'
		after.write(header)
		for line in before:
			parts = line.split(delimiter)
			print(parts)
			typ = parts[0].replace("type=", "")
			length = parts[1].replace("len=", "")
			word1 = parts[2].replace("word1=", "")
			pos1 = parts[3].replace("pos1=", "")
			stemmed1 = parts[4].replace("stemmed1=", "")
			priorpolarity = parts[5].replace("priorpolarity=", "")
			arabic = parts[6].replace("ar=", "")
			buckwalter = parts[7].strip().replace("bw=", "")
			new_line = '"{0}","{1}","{2}","{3}", "{4}", "{5}", "{6}", "{7}"'.format(typ, length, word1, pos1, stemmed1, priorpolarity, arabic, buckwalter)
			
			after.write(new_line)
			after.write("\n")

def spanishemotionlexicon_to_arff(infile, outfile, delimiter="", parts=3):
	"""first remove all instructions at the top, only provide the lines of actual data of the file """
	with open(infile, "r", encoding="utf-8") as before, open(outfile, "w", encoding="utf-8") as after:
		header='@relation '+infile+'\n\n@attribute spanish string \n@attribute emotion string\n@attribute score numeric\n\n@data\n'
		after.write(header)
		for line in before:
			print(line)
			parts = line.split()
			spanish = parts[0]
			emotion = parts[2]
			score = parts[1]
			new_line = '"{0}","{1}",{2}'.format(spanish, emotion, score)
			after.write(new_line)
			after.write("\n")

if __name__ == "__main__":
	args = create_arg_parser()
	spanishemotionlexicon_to_arff(args.f, args.o)

