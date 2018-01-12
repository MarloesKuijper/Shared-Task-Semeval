#!/bin/bash

## Do the feature extraction for all tasks here

## Names of folders (probably does not need to be changed)
EIREG="EI-Reg/"
EIOC="EI-Oc/"
VREG="V-Reg/"
VOC="V-Oc/"

## Train/dev/test folders (probably does not need to be changed)
TRAIN="train/"
DEV="dev/"
TEST="test/"

## Folders where we save all information (needs to be changed!)
DATE_FOLDER="12jan_emb10jan/"
BEST_FEATURES="../best_features/"
ALL_FEATURES="../../all_features/"
BASE="../data_for_feature_extraction/es/"

## Embedding file we use (needs to be changed!)
EMB="/net/aistaff/rikvannoord/public_html/affect_tweets/embedding_tests/jan_10/embeddings/trained_emb_mc5_s200_win45_sg0.txt.csv.gz"

## Translated files (incomplete currently)
EIREG_TRANS="-tr ../translated_tweet_data/en_to_es_arff/"
EIOC_TRANS=""
VREG_TRANS=""
VOC_TRANS=""

## Are we on Unix or not (if not leave empty "")
UNIX='-u'

## What Python do we use (I have to type python3 that's why it's added)
PYTHONTYPE="python3"

## Do test lexicons here (takes quite a while) -- also create directories that are needed

## EI-Reg
mkdir -p $BEST_FEATURES$EIREG$DATE_FOLDER
mkdir -p $ALL_FEATURES$EIREG
$PYTHONTYPE test_lexicons.py -e $BASE$EIREG$TRAIN -f $ALL_FEATURES$EIREG -b $BEST_FEATURES$EIREG$DATE_FOLDER -emb $EMB -t $BASE$EIREG$TEST -d $BASE$EIREG$DEV  $EIREG_TRANS $UNIX

## EI-Oc
mkdir -p $BEST_FEATURES$EIOC$DATE_FOLDER
mkdir -p $ALL_FEATURES$EIOC
$PYTHONTYPE test_lexicons.py -e $BASE$EIOC$TRAIN -f $ALL_FEATURES$EIOC -b $BEST_FEATURES$EIOC$DATE_FOLDER -emb $EMB -t $BASE$EIOC$TEST -d $BASE$EIOC$DEV $EIOC_TRANS $UNIX --clf

## V-Reg
mkdir -p $BEST_FEATURES$VREG$DATE_FOLDER
mkdir -p $ALL_FEATURES$VREG
$PYTHONTYPE test_lexicons.py -e $BASE$VREG$TRAIN -f $ALL_FEATURES$VREG -b $BEST_FEATURES$VREG$DATE_FOLDER -emb $EMB -t $BASE$VREG$TEST -d $BASE$VREG$DEV $VREG_TRANS $UNIX --val

## V-Oc
mkdir -p $BEST_FEATURES$VOC$DATE_FOLDER
mkdir -p $ALL_FEATURES$VOC
$PYTHONTYPE test_lexicons.py -e $BASE$VOC$TRAIN -f $ALL_FEATURES$VOC -b $BEST_FEATURES$VOC$DATE_FOLDER -emb $EMB -t $BASE$VOC$TEST -d $BASE$VOC$DEV $VOC_TRANS $UNIX --clf --val
