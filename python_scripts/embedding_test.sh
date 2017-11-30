#!/bin/bash

## Script to extract features for certain trained word embeddings + data file

#source config file to have $WEKA_HOME
source /home/p266548/Documents/affect_tweets/Shared-Task-Semeval/config.sh

#get files from command line parameters
EMBEDDING_FILE=$1
EMOTION_FILE=$2
FEATURE_FILE=$3

WEKA_JAR='weka.jar'
WEKA_CALL=$WEKA_HOME$WEKA_JAR

#Remove ID
eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$ -i $EMOTION_FILE -o tmp1.arff"

#Add embeddings
eval "java -Xmx4G -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B $EMBEDDING_FILE -S 0 -K 15 -L -O -i tmp1.arff -o tmp2.arff"

#Reorder to have correct format
eval "java -cp $WEKA_CALL weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3 -i tmp2.arff -o tmp.arff"

#Save features
eval "java -cp $WEKA_CALL weka.Run weka.core.converters.CSVSaver -i tmp.arff -o $FEATURE_FILE"
