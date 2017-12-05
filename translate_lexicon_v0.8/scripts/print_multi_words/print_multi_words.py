import sys

lexicon = sys.argv[1]
# print good lines
line_count = 0
multi_found = 0
with open(lexicon, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.split()
        line_count += 1
        if len(line) != 3:
            print("Found on line {0} {1}".format(line_count, line))
            multi_found += 1

print()
print("Total multi words found: {0}".format(multi_found))
