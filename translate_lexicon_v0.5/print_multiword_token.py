# print good lines
count = 0
with open("NRC-emotion.txt", "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.split()
        if len(line) == 3:
            print(line)
            count += 1

print(count)
