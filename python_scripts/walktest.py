import os
filenames = []
for dirpath, dirnames, files in os.walk('../features'):
    for name in files:
        if name not in ".DS_Store":
            print(name)
            filenames.append(os.path.join(dirpath, name))
excel_data = []
