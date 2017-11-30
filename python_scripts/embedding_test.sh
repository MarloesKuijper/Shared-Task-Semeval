#!bin/bash

EMBEDDING_FILE=$1
EMOTION_FILE=$2
FEATURE_FILE=$3

echo $EMBEDDING_FILE
echo $EMOTION_FILE
echo $FEATURE_FILE
# step 1: remove id
# REMOVE_ID="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$"
# EMBEDDINGS_ES="java -Xmx4G -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B C:/Users/marlo/Documents/school/master/shared_task/wordembeddings/Scraped_tok_embeddings.csv.gz -S 0 -K 15 -L -O"
# # step 3: reorder
# REORDER="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3"

# # step 4: save
# SAVE="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.core.converters.CSVSaver -i tmp.arff -o"

eval "java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$ -i $EMOTION_FILE -o tmp1.arff"
eval "java -Xmx4G -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B $EMBEDDING_FILE -S 0 -K 15 -L -O -i tmp1.arff -o tmp2.arff"
eval "java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3 -i tmp2.arff -o tmp.arff"
eval "java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.core.converters.CSVSaver -i tmp.arff -o $FEATURE_FILE"