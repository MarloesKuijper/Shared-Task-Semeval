import subprocess
import os
import sys
from bs4 import UnicodeDammit
import time

# print good lines
count = 0
with open("NRC-emotion.txt", "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.split()
        if len(line) == 3:
            print(line)
            count += 1

print(count)

# clean up
with open("NRC-emotion.txt", "r", encoding="utf-8") as infile:
    with open("NRC-emotion-cleaned.txt", "w", encoding="utf-8") as outfile:
        for line in infile:
            write_line = line
            line = line.split()

            if len(line) == 3:
                print(line)
                count += 1
                outfile.write(write_line.rstrip() + "\n")  # write line to new file
