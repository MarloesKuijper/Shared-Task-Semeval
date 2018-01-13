# -*- coding: utf-8 -*-
import subprocess
import os
import sys
from bs4 import UnicodeDammit
import time

def find_files(directory, lexicon):
    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(lexicon)]


def replace_tokens_in_original_lexicon(lexicon, directory, target_file, token_position, input_token_file):
    print("Reading lexicon: {0}....".format(lexicon))

    token_position = int(token_position)

    if input_token_file != "":
        input_tokens = []
        with open(input_token_file, "r", encoding="utf-8") as read_tokens:
            for line in read_tokens:
                input_tokens.append(line)


    for fl in find_files(directory, lexicon):
        with open(fl, "r", encoding="utf-8") as infile:
            with open(target_file, "w", encoding="utf-8") as outfile:  # open file to write
                for index, line in enumerate(infile):

                    # tokens = line.split('","')
                    # #print(tokens[2])
                    # tokens[0] = tokens[0] + '","'
                    # #print(tokens[0])
                    # tokens[2] = '","' + tokens[2]
                    #print(tokens[2])
                    #print(line)

                    tokens = line.split('\t')
                    ID = tokens[0]
                    label = tokens[-2]
                    label2 = tokens[-1]
                    token = tokens[1:-2]
                    token = token[0]

                    #print(ID.encode("utf-8"))
                    #print(token.encode("utf-8"))

                    if replace_tokens_in_original_lexicon:
                        new_token = input_tokens[index]
                        new_token = new_token.rstrip()
                        #new_token = new_token

                        print(index)
                        #print(token_position)

                        #print(new_token.encode("utf-8"))

                        #print(len(new_token))
                        #print(tokens)


                        #print("HOI")
                        new_line = str(ID) + "\t" + str(new_token) + "\t" + str(label) + '\t' + str(label2)  #

                        #print(new_line.encode("utf-8"))



                        # uncomment when there is need for more posistion, again it is lexicon dependent
                        # if token_position == 3:
                        #     new_line = str(tokens[0]) + "\t" + str(tokens[1]) + "\t" + str(tokens[3]) + "\t" + str(new_token)
                        #
                        # if token_position == 4:
                        #     new_line = str(tokens[0]) + "\t" + str(tokens[1]) + "\t" + str(tokens[3]) + "\t" + str(tokens[4])+ "\t" + str(new_token)

                        # write desired output to target_file
                        outfile.write(new_line)  # write line to new file

def main():
    # just a timer to benchmark the system
    t0 = time.time()

    # argument control / start the process
    if len(sys.argv) != 5:
        print("Usage: replace_tokens.py  <lexicon>    <target_file>  <token_position_to_replace>  <token_input_list>")  # token position starts from 0!
        print()
        print("Usage: python3 bash.py NRC-emotion-lexicon-wordlevel-v0.92.txt NRC-translated.txt 0 tokens.txt")
        sys.exit (1)
    else:
        replace_tokens_in_original_lexicon(sys.argv[1], "./", sys.argv[2], sys.argv[3], sys.argv[4])

    # output system runtime
    test_time = time.time() - t0
    print("Runtime: ", test_time)


if __name__ == '__main__':
    main()
