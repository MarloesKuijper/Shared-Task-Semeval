#!bin/bash

# NOTE: all files have been normalized > lowercased file names, delimited by - not _, with 2018 at the beginning of each filename

# change these if necessary (and change WEKA-3-8 in script below if necessary)
SHARED_TASK_FOLDER=C:/Users/marlo/Documents/school/master/shared_task/
REPO_FOLDER=C:/Users/marlo/Documents/school/master/shared_task/repo/

# don't change these
DATA_FOLDER=~/Documents/school/master/shared_task/repo/data_for_feature_extraction/
FEATURE_FOLDER=~/Documents/school/master/shared_task/repo/features/

FILES=$(find $DATA_FOLDER -type f)

FEATURE_NAME="ngrams.csv"

for item in $FILES
do
	cd $REPO_FOLDER
	FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
	LANG=`echo $FILE_NAME | cut -d' ' -f 1`
	EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
	TYPE=`echo $FILE_NAME | cut -d' ' -f 3`

	cd features/$LANG/$EMOTION/$TYPE

	# step 1: remove id
 	java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$ -i $item -o $SHARED_TASK_FOLDER/tmp.arff

	# step 2: add filter tweettosparse > change output work in progress
	java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToSparseFeatureVector -M 0 -I 1 -Q 1 -D 3 -E 5 -L -F -G 0 -I 0 -i $SHARED_TASK_FOLDER/tmp.arff -o $SHARED_TASK_FOLDER/tmp2.arff

	# step 3: reorder
	java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3 -i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp.arff

	# step 4: save
	java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.core.converters.CSVSaver -i $SHARED_TASK_FOLDER/tmp.arff -o ./$FEATURE_NAME

done

