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


def save_tokens_from_original_lexicon(lexicon, target_file, token_position):
    print("Reading lexicon: {0}....".format(lexicon))

    token_position = int(token_position)

    for fl in find_files("./", lexicon):
        with open(fl, "r", encoding="utf-8") as infile:
            with open(target_file, "w", encoding="utf-8") as outfile:  # open file to write

                for index, line in enumerate(infile):
                    # format of input file
                    #########################################################################################
                    #   Warning!                                                                            #
                    #   The position of token(s) to be translated depends on the original lexicon format    #
                    tokens = line.split('","')
                    print(tokens)                                                        #
                    token = tokens[token_position]                                                                       #
                    #                                                                                       #
                    #########################################################################################

                    new_line = str(token)

                    # write desired output to target_file
                    outfile.write(new_line + "\n")  # write line to new file


def main():
    # just a timer to benchmark the system
    t0 = time.time()

    # argument control / start the process
    if len(sys.argv) != 4:
        print("Usage: save_tokens.py <lexicon> <target_file> <token_position> ")  # token position starts from 0!
        print()
        print("Usage: python3 bash.py NRC-emotion-lexicon-wordlevel-v0.92.txt NRC-translated.txt 0")
        sys.exit (1)
    else:
        save_tokens_from_original_lexicon(sys.argv[1], sys.argv[2], sys.argv[3])

    # output system runtime
    test_time = time.time() - t0
    print("Runtime: ", test_time)


if __name__ == '__main__':
    main()
