import subprocess
import os
import sys
from bs4 import UnicodeDammit
import time

def find_files(directory, lexicon, extension):
    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(lexicon+"."+extension)]


def translate_lexicon(lexicon, directory, extension, target_file, target_extension, input_token_file):
    unicode_chars = '\\'
    print("Reading lexicon: {0}....".format(lexicon))

    if input_token_file != "":
        input_tokens = []
        with open(input_token_file, "r", encoding="utf-8") as read_tokens:
            for line in read_tokens:
                input_tokens.append(line)


    # settings.....
    # Option True for translate / False for storing original tokens
    do_translate = False
    save_tokens_from_original_lexicon = False
    replace_tokens_in_original_lexicon = True

    for fl in find_files(directory, lexicon, extension):
        with open(fl, "r", encoding="utf-8") as infile:
            with open(target_file+"."+target_extension, "w", encoding="utf-8") as outfile:  # open file to write
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

                    if replace_tokens_in_original_lexicon:
                        # print("HOI")
                        # print(input_tokens[index])

                        new_token = input_tokens[index]
                        new_token = new_token.rstrip()

                        voorzetsels_spaans = 'de '

                        if 'de ' in new_token or 'la ' in new_token or 'las ' in new_token:
                            print(new_token)
                            new_token = new_token[3:]
                            print(new_token)


                        new_line = str(new_token) + "\t" + str(tokens[1]) + "\t" + str(tokens[2])  #
                        # print(token, new_line)
                        # print("EN {0} -> ES {1}".format(token, new_token, new_line))

                        # write desired output to target_file
                        outfile.write(new_line + "\n")  # write line to new file


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


                    # when do_translate = False
                    if save_tokens_from_original_lexicon:
                        new_line = str(token)

                        # write desired output to target_file
                        outfile.write(new_line + "\n")  # write line to new file

        # print multi token info
        print("multi_tokens: {0}".format(multi_token_list))
        print("multi_tokens found: {0}".format(count_multi_token))

def main():

    if sys.argv[5]:
        input_token_file = sys.argv[5]
    else:
        input_token_file = ""

    # just a timer to benchmark the system
    t0 = time.time()

    # argument control / start the process
    if len(sys.argv) != 6:
        print("Usage: python3 <program> <lexicon> <extension> <target_file> <target_extension>")
        print("Usage: python3 bash.py NRC-emotion-lexicon-wordlevel-v0.92 .txt NRC-translated .txt")
        sys.exit (1)
    else:
        print("hoi")
        translate_lexicon(sys.argv[1], "./", sys.argv[2], sys.argv[3], sys.argv[4], input_token_file)

    # output system runtime
    test_time = time.time() - t0
    print("Translate time: ", test_time)


if __name__ == '__main__':
    main()
