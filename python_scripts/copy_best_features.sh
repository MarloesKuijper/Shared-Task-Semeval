#!/bin/bash

#get files from command line parameters
TARGET_FOLDER=$1
ANGER=$2
FEAR=$3
JOY=$4
SADNESS=$5

echo "starting to copy best feature files to target dir..."

cp "$ANGER" "$TARGET_FOLDER"

cp "$FEAR" "$TARGET_FOLDER"

cp "$JOY" "$TARGET_FOLDER"

cp "$SADNESS" "$TARGET_FOLDER"

echo "successfully copied best feature files to target dir"