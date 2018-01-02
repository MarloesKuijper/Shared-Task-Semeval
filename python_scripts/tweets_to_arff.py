# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# tweets_to_arff.py
# felipebravom
# Running example: python tweets_to_arff data/anger-ratings-0to1.test.target.tsv data/anger-ratings-0to1.test.target.arff
import sys, os, re


def create_arff(input_file,output_file):
    """
    Creates an arff dataset
    """




    out=open(output_file,"w", encoding="utf-8")
    header='@relation '+ '"' + input_file+ '"' + '\n\n@attribute id string \n@attribute tweet string\n@attribute emotion string\n@attribute score numeric \n\n@data\n'
    out.write(header)



    f=open(input_file, "r", encoding="utf-8")
    lines=f.readlines()


    for line in lines[1:]:
        parts=line.split("\t")
        if len(parts)==4:

            id=parts[0]
            tweet=parts[1]
            emotion=parts[2]
            score=parts[3].strip()
            score = score if score != "NONE" else "?"

            out_line='"{0}","{1}","{2}", {3}'.format(id, tweet, emotion, score)
            out.write(out_line)
            out.write("\n")
        else:
            print("Wrong format")


    f.close()
    out.close()


if __name__ == "__main__":
    # find files in certain directory that need to be converted
    file_dir = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]
    create_arff(infile, outfile)
