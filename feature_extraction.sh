#!bin/bash

# change these if necessary (and change WEKA-3-8 in script below if necessary)
SHARED_TASK_FOLDER=C:/Users/marlo/Documents/school/master/shared_task
REPO_FOLDER=C:/Users/marlo/Documents/school/master/shared_task/repo/

# don't change these
DATA_FOLDER=~/Documents/school/master/shared_task/repo/data_for_feature_extraction
FEATURE_FOLDER=~/Documents/school/master/shared_task/repo/features/

#FILES_EN=$(find $DATA_FOLDER/en -type f)
FILES_ES=$(find $DATA_FOLDER/es -type f)

# step 1: remove id
REMOVE_ID="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$"

# step 2: add filter tweettosparse > change output work in progress
NGRAMS="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToSparseFeatureVector -M 0 -I 1 -Q 1 -D 3 -E 5 -L -F -G 0 -I 0 -i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
LEXICONS_EN="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 1 -A -D -F -H -J -L -N -P -Q -R -T -U -O"
LEXICONS_ES="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:\\Users\\marlo\\wekafiles\\packages\\AffectiveTweets\\lexicons\\spanishemotionlexicon.arff -B SpanishEmotionLex -A 1\" -I 1 -U"
#LEXICONS_AR="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/wekafiles/packages/AffectiveTweets/lexicons/arabic/SemEval2016-Arabic-Twitter-Lexicon/SemEval2016-Arabic-Twitter-Lexicon_adjusted.arff -B ArabicSemevalLex -A 1\" -I 1 -U"

# deze gebruiken met meerdere lexicons tegelijk (naam veranderen (2 weghalen)) BETA
# LEXICONS_AR_COMBINE="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run -F weka.filters.MultiFilter weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/wekafiles/packages/AffectiveTweets/lexicons/arabic/SemEval2016-Arabic-Twitter-Lexicon/SemEval2016-Arabic-Twitter-Lexicon_adjusted.arff -B ArabicSemevalLex -A 1\" -I 1 -U \
#weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/wekafiles/packages/AffectiveTweets/lexicons/arabic/SemEval2016-Arabic-Twitter-Lexicon/list-Arabic-negators.arff -B ArabicNegations -A 1\" -I 1 -U"

# LEXICONS_AR_BL="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/Documents/school/master/shared_task/repo/lexicons/arabic/translated_from_english/raw+arff/bingliu_ar.arff -B bingliu -A 1\" -I 1 -U"

# LEXICONS_AR_MPQA="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/Documents/school/master/shared_task/repo/lexicons/arabic/translated_from_english/raw+arff/MPQA_ar.arff -B mpqa -A 1\" -I 1 -U"

# LEXICONS_AR_S140="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/Documents/school/master/shared_task/repo/lexicons/arabic/translated_from_english/raw+arff/S140-unigrams-pmilexicon_ar.arff -B s140 -A 1\" -I 1 -U"

# LEXICONS_AR_NRC_HS="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/Documents/school/master/shared_task/repo/lexicons/arabic/translated_from_english/raw+arff/NRC-HS-unigrams-pmilexicon_ar.arff -B nrc_hashtag -A 1\" -I 1 -U"

# LEXICONS_AR_NRC_EMO="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/Documents/school/master/shared_task/repo/lexicons/arabic/translated_from_english/raw+arff/nrc_emotion_ar.arff -B nrc_emotion -A 1\" -I 1 -U"


EMBEDDINGS_EN="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B C:/Users/marlo/wekafiles/packages/AffectiveTweets/resources/w2v.twitter.edinburgh.100d.csv.gz -S 0 -K 15 -L -O"
EMBEDDINGS_ES="java -Xmx4G -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B C:/Users/marlo/Documents/school/master/shared_task/wordembeddings/Scraped_tok_embeddings.csv.gz -S 0 -K 15 -L -O"
EMBEDDINGS_AR="java -Xmx4G -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B C:/Users/marlo/Documents/school/master/shared_task/wordembeddings/wiki.ar_reordered_test.csv.gz -S 0 -K 15 -L -O"


# step 3: reorder
REORDER="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.Reorder -R 4-last,3"

# step 4: save
SAVE="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.core.converters.CSVSaver -i $SHARED_TASK_FOLDER/tmp.arff -o"

function feat_extractor {
	#echo $5
	case $2 in
		1 )
			echo "option1: ngrams"
			
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="ngrams.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval $NGRAMS
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				eval $SAVE "./$FEATURE_NAME"
			done ;;

		2 )
			echo "option 2: lexicons"
			# select the right lexicon for the language in question
			lang=`echo $1 | tr [a-z] [A-Z]`
			LEXICONS=LEXICONS_${lang}
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="nrc_emotion.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval ${!LEXICONS} "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
				echo "reorder coming up"
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				eval $SAVE "./$FEATURE_NAME"
			done;;
			
		3 )
			echo "option 3: word embeddings"
			# select the right lexicon for the language in question
			lang=`echo $1 | tr [a-z] [A-Z]`
			EMBEDDINGS=EMBEDDINGS_${lang}
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="embeddings_espana_1.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval ${!EMBEDDINGS} "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				eval $SAVE "./$FEATURE_NAME"
			done;;
		4 )
			echo "option 4: 1 + 2 + 3"
			lang=`echo $1 | tr [a-z] [A-Z]`
			LEXICONS=LEXICONS_${lang}
			EMBEDDINGS=EMBEDDINGS_${lang}
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="ngrams-lexicons-embeddings.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval $NGRAMS
				echo "lexicons"
				eval ${!LEXICONS} "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp1.arff"
				echo "embeddings"
				eval ${!EMBEDDINGS} "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
				echo "reorder"
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				echo "save"
				eval $SAVE "./$FEATURE_NAME"
			done
			;;
		5 )
			echo "option 5: 1 + 2"
			lang=`echo $1 | tr [a-z] [A-Z]`
			LEXICONS=LEXICONS_${lang}
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="ngrams-lexicons.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval $NGRAMS
				echo "lexicons"
				eval ${!LEXICONS} "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp1.arff"
				echo "reorder"
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				echo "save"
				eval $SAVE "./$FEATURE_NAME"
			done
			;;
		6 )
			echo "option 6: 1 + 3" 
			lang=`echo $1 | tr [a-z] [A-Z]`
			EMBEDDINGS=EMBEDDINGS_${lang}
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="ngrams-embeddings.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval $NGRAMS
				echo "embeddings"
				eval ${!EMBEDDINGS} "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp1.arff"
				echo "reorder"
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				echo "save"
				eval $SAVE "./$FEATURE_NAME"
			done
			;;
		7 )
			echo "option 7: 2 + 3"
			lang=`echo $1 | tr [a-z] [A-Z]`
			LEXICONS=LEXICONS_${lang}
			EMBEDDINGS=EMBEDDINGS_${lang}
			for item in $3
			do
				echo $item
				cd $REPO_FOLDER
				FEATURE_NAME="lexicons-embeddings.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				cd features/$1/$EMOTION/$TYPE
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				echo "lexicons"
				eval ${!LEXICONS} "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
				echo "embeddings"
				eval ${!EMBEDDINGS} "-i $SHARED_TASK_FOLDER/tmp2.arff -o $SHARED_TASK_FOLDER/tmp1.arff"
				echo "reorder"
				eval $REORDER "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp.arff"
				echo "save"
				eval $SAVE "./$FEATURE_NAME"
			done
			;;
	esac

}


feat_extractor "es" 3 "$FILES_ES"

