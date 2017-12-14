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


def clean_lexicon(directory, lexicon, target_file):
    print("Cleaning........")
    for fl in find_files("./", lexicon):
        with open(fl, "r", encoding="utf-8") as infile:
            with open(target_file, "w", encoding="utf-8") as outfile:  # open file to write

                for line in infile:
                    # remove other multi tokens and write new token to file
                    if len(line.split()) == 3:
                        outfile.write(line.rstrip() + "\n")  # write line to new file


def main():
    # just a timer to benchmark the system
    t0 = time.time()

    # argument control / start the process
    if len(sys.argv) != 3:
        print("Usage: clean_multi_words.py <lexicon> <target>")  # token position starts from 0!
        print()
        print("Usage: python3 clean_multi_words.py NRC-lexicon.txt NRC-lexicon_cleaned.txt")
        print()
        sys.exit (1)
    else:
        clean_lexicon("./", sys.argv[1], sys.argv[2])

    # output system runtime
    test_time = time.time() - t0
    print("Runtime: ", test_time)


if __name__ == '__main__':
    main()
