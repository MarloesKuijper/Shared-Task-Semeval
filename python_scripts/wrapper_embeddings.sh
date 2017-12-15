#!/bin/bash

### Script that runs pipeline for training + evaluating sets of word embeddings (can take a long time) ###


## Get files from command line parameters

## Directory with all tweets, read all files ending in .txt
TWEET_FOLDER=$1

## Directory with emotion data
EMOTION_FOLDER=$2

## Root of test -- in this folder create embeddings/features
TEST_ROOT=$3
FEATURE_NAME='feature_files/'
EMBEDDING_NAME='embeddings/'

## Directory to save extracted features to (will create if doesn't exist)
FEATURE_FOLDER=$TEST_ROOT$FEATURE_NAME

## Directory to save trained embeddings to (will create if doesn't exist)
EMBEDDING_FOLDER=$TEST_ROOT$EMBEDDING_NAME

## Number of workers (parallel threads I think, is obviously lower if you run on your own CPU!)
WORKERS=12

##Create dirs if they do not exist
mkdir -p $FEATURE_FOLDER
mkdir -p $EMBEDDING_FOLDER

## Run train embedding script -- check python script for parameter settings
## Note that OMP_NUM_THREADS is necessary for me, for some reason if I do not add that it fills our whole server
## You can (have to) remove this if you want to run this version 
OMP_NUM_THREADS=$WORKERS python train_embeddings_multi.py -d $TWEET_FOLDER -o $EMBEDDING_FOLDER -p $WORKERS

## Now test the embeddings with a default SVM to find the best settings, save results
## Remove -u at the end if you run Windows
python test_embeddings.py -e $EMOTION_FOLDER -w $EMBEDDING_FOLDER -f $FEATURE_FOLDER -u
