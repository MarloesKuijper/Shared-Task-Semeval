################

!!!!!! ALL FILES NEED TO BE PLACED IN THE SAME DIRECTORY !!!!!!!

################
Process scripts:
save_tokens.py:
Usage: save_tokens.py <input_lexicon> <write_to_target_file> <token_position_to_scrape> 

replace_tokens.py:
Usage: replace_tokens.py  <input_lexicon> <target_file> <token_position_to_replace> <token_input_list>

clean_multi_words.py:
Usage: clean_multi_words.py <lexicon> <target>

################
All in one:
translate.py:
Usage: python3 <program> <lexicon> <target_file>
python3 translate.py NRC-emotion-lexicon-wordlevel-v0.92.txt NRC-translated.txt

################
Multi word scripts:
print_multiword_token.py:
python3 print_multi_words.py vies.txt 
Found on line 1 ['outraged', 'hello', '0.964', 'anger']
Found on line 12 ['brutality', 'hi', '0.959', 'anger']

clean_tokens.py:
Clean multi_tokens. See rules in script
python3 clean_tokens.py vies.txt schoon.txt

################
Workflow Translate a lexicon using Apertium's website (see example below):

1.  Pick the lexicon you want to translate. 
2.  run script save_tokens_from_original_lexicon to gather the tokens you want to translate (step 1) 
3.  translate the document on apertium's website
4.  replace the translated tokens on the position of the original using replace_tokens..... (step 2)
5.  when multi_word_expressions were created by apertium, clean up the lexicon and write a cleaned version of the lexicon (step 3)
6.  compress the clean version of the lexicon into .gz 
7.  put the archived file into your affective tweet package 

8.  Run the tweettolexiconfeature in Weka 
9.  up to you 


################
Example:
1. python3 save_token_from_original_lexicon.py NRC-emotion-lexicon-wordlevel-v0.92.txt tokens_nrc.txt 0

2. python3 replace_tokens_from_original_lexicon.py NRC-emotion-lexicon-wordlevel-v0.92.txt replaced_nrc.txt 0 tokens_nrc_ES.txt

3. python3 clean_multitokens_in_lexicon.py replaced_nrc.txt replaced_nrc_cleaned.txt


