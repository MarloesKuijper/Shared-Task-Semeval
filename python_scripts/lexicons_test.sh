#!/bin/bash

## Script to extract features for certain trained word embeddings + data file

#source config file to have $WEKA_HOME
source ../../config.sh

#get files from command line parameters
EMOTION_FILE=$1
FEATURE_FILE=$2
EMBEDDING_FILE=$3
SELECTION=${@:4}
SELECTION_W_SS=${@:5}


WEKA_JAR='weka.jar'
WEKA_CALL=$WEKA_HOME$WEKA_JAR


echo "$2"


if [ $4 == "sentistrength" ];
then
	if [ -n "$5" ];
	then
		#Remove ID
		eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$ -i $EMOTION_FILE -o tmp1.arff"

		#Add embeddings
		eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B $EMBEDDING_FILE -S 0 -K 15 -L -O -i tmp1.arff -o tmp2.arff"

		#Add sentilexicons
		eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToSentiStrengthFeatureVector -i tmp2.arff -o tmp3.arff"

		#Add lexicons
		eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 1 $SELECTION_W_SS -U -i tmp3.arff -o tmp2.arff"

		#Reorder to have correct format
		eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3 -i tmp2.arff -o tmp.arff"

		#Save features
		eval "java -cp $WEKA_CALL weka.Run weka.core.converters.CSVSaver -i tmp.arff -o $FEATURE_FILE"
	else

		echo "only sentistrength selected"
		#Remove ID
		eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$ -i $EMOTION_FILE -o tmp1.arff"

		#Add embeddings
		eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B $EMBEDDING_FILE -S 0 -K 15 -L -O -i tmp1.arff -o tmp2.arff"

		#Add lexicons
		eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToSentiStrengthFeatureVector -i tmp2.arff -o tmp3.arff"

		#Reorder to have correct format
		eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3 -i tmp3.arff -o tmp.arff"

		#Save features
		eval "java -cp $WEKA_CALL weka.Run weka.core.converters.CSVSaver -i tmp.arff -o $FEATURE_FILE"
	fi
else
	echo "other selected"
	#Remove ID
	eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$ -i $EMOTION_FILE -o tmp1.arff"

	#Add embeddings
	eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B $EMBEDDING_FILE -S 0 -K 15 -L -O -i tmp1.arff -o tmp2.arff"

	#Add lexicons
	eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 1 $SELECTION -U -i tmp2.arff -o tmp3.arff"

	#Reorder to have correct format
	eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3 -i tmp3.arff -o tmp.arff"

	#Save features
	eval "java -cp $WEKA_CALL weka.Run weka.core.converters.CSVSaver -i tmp.arff -o $FEATURE_FILE"
fi
