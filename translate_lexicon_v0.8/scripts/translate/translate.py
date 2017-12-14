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


def translate_lexicon(lexicon, directory, target_file):
    unicode_chars = '\\'
    print("Reading lexicon: {0}....".format(lexicon))

    # settings.....
    do_translate = True

    for fl in find_files(directory, lexicon):
        with open(fl, "r", encoding="utf-8") as infile:
            with open(target_file, "w", encoding="utf-8") as outfile:  # open file to write
                print("Translating......")

                # set multi_token counter and list
                count_multi_token = 0
                multi_token_list = []

                for index, line in enumerate(infile):
                    # format of input file
                    #########################################################################################
                    #   Warning!                                                                            #
                    #   The position of token(s) to be translated depends on the original lexicon format    #
                    tokens = line.split()                                                                   #
                    token = tokens[0]                                                                       #
                    #                                                                                       #
                    #########################################################################################

                    # translate process
                    if do_translate:
                        command = "echo "+token+" | apertium en-es"
                        translated = subprocess.Popen(str(command), stdout=subprocess.PIPE, shell=True).stdout.read()
                        translated_clean = translated.split()
                        translated_clean = str(translated_clean)
                        translated_clean = translated_clean[3:-2]

                        # clean up unicode
                        dammit = UnicodeDammit(translated)
                        translated_clean = dammit.unicode_markup
                        translated_clean = translated_clean.rstrip()

                        # take last token of multi_token
                        if len(translated_clean.split()) > 1:
                            multi_token_list.append(translated_clean)
                            translated_clean = str(translated_clean.split()[-1])
                            count_multi_token += 1

                        # remove asterix when translation failed
                        translated_clean_star = ""
                        for char in translated_clean:
                            if char == "*":
                                translated_clean_star = translated_clean_star + ""
                            else:
                                translated_clean_star = translated_clean_star + char

                        # just give it a better name
                        translated_token = translated_clean_star

                        # format of output file
                        ###########################################################################################
                        #   Warning!                                                                              #
                        #   Change output format equal to original lexicon format                                 #
                        new_line = str(translated_token.lower()) + "\t" + str(tokens[1]) + "\t" + str(tokens[2])  #
                        print("EN {0} -> ES {1}".format(token, new_line))                                         #
                        #                                                                                         #
                        ###########################################################################################

                        # write desired output to target_file
                        outfile.write(new_line + "\n")  # write line to new file


        # print multi token info
        print("multi_tokens: {0}".format(multi_token_list))
        print("multi_tokens found: {0}".format(count_multi_token))

def main():
    # just a timer to benchmark the system
    t0 = time.time()

    # argument control / start the process
    if len(sys.argv) != 3:
        print("Usage: python3 <program> <lexicon> <target_file>")
        print("Usage: python3 translate.py NRC-emotion-lexicon-wordlevel-v0.92.txt NRC-translated.txt")
        sys.exit (1)
    else:
        translate_lexicon(sys.argv[1], "./", sys.argv[2])

    # output system runtime
    test_time = time.time() - t0
    print("Translate time: ", test_time)


if __name__ == '__main__':
    main()
