
#!bin/bash

# change these if necessary (and change WEKA-3-8 in script below if necessary)
SHARED_TASK_FOLDER=C:/Users/Mike/Desktop/Shared-Task-Semeval
REPO_FOLDER=C:/Users/Mike/Desktop/Shared-Task-Semeval

# don't change these
DATA_FOLDER=~/Desktop/Shared-Task-Semeval/data_for_feature_extraction
FEATURE_FOLDER=~/Desktop/Shared-Task-Semeval/features

FILES_EN=$(find $DATA_FOLDER/en -type f)
FILES_ES=$(find $DATA_FOLDER/es -type f)
FILES_AR=$(find $DATA_FOLDER/ar -type f)


TEST_LEXICON_PATH="C:\\Users\\Mike\\Desktop\\Shared-Task-Semeval\\lexicons\\spanish\\ElhPolar_esV1proc.arff"

# step 1: remove id
REMOVE_ID="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.RemoveByName -E ^.*id$"

# step 2: add filter tweettosparse > change output work in progress
NGRAMS="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToSparseFeatureVector -M 0 -I 1 -Q 1 -D 3 -E 5 -L -F -G 0 -I 0 -i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
# LEXICONS_EN="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 1 -A -D -F -H -J -L -N -P -Q -R -T -U -O"
LEXICONS_ES="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile "$TEST_LEXICON_PATH" -B SpanishEmotionLex -A 1\" -I 1 -U"
# LEXICONS_AR="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/wekafiles/packages/AffectiveTweets/lexicons/arabic/SemEval2016-Arabic-Twitter-Lexicon/SemEval2016-Arabic-Twitter-Lexicon_adjusted.arff -B ArabicSemevalLex -A 1\" -I 1 -U"

# deze gebruiken met meerdere lexicons tegelijk (naam veranderen (2 weghalen))
LEXICONS_AR2="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run -F weka.filters.MultiFilter weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/wekafiles/packages/AffectiveTweets/lexicons/arabic/SemEval2016-Arabic-Twitter-Lexicon/SemEval2016-Arabic-Twitter-Lexicon_adjusted.arff -B ArabicSemevalLex -A 1\" -I 1 -U \
weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -O -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile C:/Users/marlo/wekafiles/packages/AffectiveTweets/lexicons/arabic/SemEval2016-Arabic-Twitter-Lexicon/list-Arabic-negators.arff -B ArabicNegations -A 1\" -I 1 -U"

EMBEDDINGS_EN="java -cp /c/Program\ Files/Weka-3-8/weka.jar weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 1 -B C:/Users/marlo/wekafiles/packages/AffectiveTweets/resources/w2v.twitter.edinburgh.100d.csv.gz -S 0 -K 15 -L -O"

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
				FEATURE_NAME="lexicons.csv"
				FILE_NAME=`echo $item | sed 's/.\{5\}$//' | xargs -d '-' -n3 | tail -2 | sed '/^$/d'`
				EMOTION=`echo $FILE_NAME | cut -d' ' -f 2`
				TYPE=`echo $FILE_NAME | cut -d' ' -f 3`
				# cd features/$1/$EMOTION/$TYPE = de oude goede
				cd features/$1/
				eval $REMOVE_ID "-i $item -o $SHARED_TASK_FOLDER/tmp1.arff"
				eval ${!LEXICONS} "-i $SHARED_TASK_FOLDER/tmp1.arff -o $SHARED_TASK_FOLDER/tmp2.arff"
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
				FEATURE_NAME="embeddings.csv"
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

# function: parameter 1 = language abbreviation, parameter 2 = option (pick option between 1 and 7, see function), 3 = files for that language
# you can change the language here: change both the first and third parameter (first == language, third == files for that language)
# right now only ngrams work for arabic (lexicon has a bug somehow) and for spanish only ngrams and lexicon and ngram + lexicon
feat_extractor "es" 2 "$FILES_ES"


