import subprocess
import os
import sys
from bs4 import UnicodeDammit
import time

def find_files(directory, input_token_file):
    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(input_token_file)]


def save_tokens_from_original_lexicon(directory, input_token_file, target_file):
    print("Cleaning........")

    if input_token_file != "":
        input_tokens = []
        with open(input_token_file, "r", encoding="utf-8") as read_tokens:
            for line in read_tokens:
                input_tokens.append(line)


    for fl in find_files("./", input_token_file):
        with open(fl, "r", encoding="utf-8") as infile:
            with open(target_file, "w", encoding="utf-8") as outfile:  # open file to write

                for index, line in enumerate(infile):
                    # format of input file
                    #########################################################################################
                    #   Warning!                                                                            #
                    #   The position of token(s) to be translated depends on the original lexicon format    #
                                                                #
                    #                                                                                       #
                    #########################################################################################

                    new_token = input_tokens[index]
                    new_token = new_token.rstrip()

                    print(new_token.split())

                    # remove de, la and las
                    if 'de ' in new_token or 'la ' in new_token or 'las ' in new_token:
                        # print("Before", new_token)
                        new_token = new_token[3:]
                        # print("After", new_token)


                    # remove other multi tokens and write new token to file
                    if len(new_token.split()) == 1:
                        print(new_token)
                        outfile.write(new_token.rstrip() + "\n")  # write line to new file


def main():
    # just a timer to benchmark the system
    t0 = time.time()

    # argument control / start the process
    if len(sys.argv) != 3:
        print("Usage: save_tokens_from_original_lexicon.py <input_token_file> <target_file>")  # token position starts from 0!
        print()
        print("Usage: python3 bash.py NRC-Tokens.txt NRC-tokens_cleaned.txt")
        print()
        sys.exit (1)
    else:
        save_tokens_from_original_lexicon("./", sys.argv[1], sys.argv[2])

    # output system runtime
    test_time = time.time() - t0
    print("Runtime: ", test_time)


if __name__ == '__main__':
    main()
