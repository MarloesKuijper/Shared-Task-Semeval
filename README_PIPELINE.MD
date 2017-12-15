Test on test set:
1. Remove additional quotes (remove_quotes.py)
2. Turn ids (first column) into ints (id_to_int.py)
3. Convert to arff
4. Preprocessing 
5. Run best system (best classifier/model)
6. Outputs need to be saved to the proper format (best is to use the original test file (which does not have the labels) and just add a column with the predicted labels)

Test embeddings:
./wrapper_embeddings.sh TWEET_FOLDER/   ../data_for_feature_extraction/es/    EXPERIMENT_FOLDER/

Test lexicons:
text_lexicons.py -e ../data_for_feature_extraction/es -f LEXICON_FEATURE_FOLDER (waar je ze wilt opslaan) -emb embeddings_file [--no_extract --unix]



ROADMAP
1. When all lexicons are working and embeddings are all trained, we decide upon the best combination of lexicons and embeddings (using the scripts mentioned above)
2. Use this combination to optimize the settings for the models we're using (LSTMs, ENSEMBLES, SVM). PLEASE ALSO WRITE DOWN WHAT YOU TRY WHEN YOU TRY PARAMETERS. FOR REPORT!!
3. Test embeddings for valence task, use best embeddings to test optimal combination of lexicons with those embeddings
4. Also the classification tasks (EI-OC and V-OC), but for these we use the same settings I think? Just a matter of converting regression score into one of the categories
5. Potentially: the emotion classification task OR scrape some tweets for English and do the same as for Spanish (best embeddings, best lexicons etc.)

