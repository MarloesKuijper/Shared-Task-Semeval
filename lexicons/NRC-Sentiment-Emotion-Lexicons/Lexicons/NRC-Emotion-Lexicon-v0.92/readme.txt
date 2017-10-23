NRC Word-Emotion Association Lexicon (NRC Emotion Lexicon)
Version 0.92
10 July 2011
Copyright (C) 2011 National Research Council Canada (NRC)
************************************************************


Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)


************************************************************
Accessing the NRC Sentiment and Emotion Lexicons
************************************************************

Since February 16, 2017 this lexicon is no longer available for direct download at this website: http://saifmohammad.com/WebPages/lexicons.html. It is still available free of charge for research purposes. It is also available for commercial applications via a perpetual commercial license for a nominal one-time fee. 

Simply email:

Saif M. Mohammad (Senior Research Officer at NRC and creator of the lexicon): Saif.Mohammad@nrc-cnrc.gc.ca 
and 
Pierre Charron (Client Relationship Leader at NRC): Pierre.Charron@nrc-cnrc.gc.ca

and include:

- your name and affiliation
- the name of the resource you are interested in
- what you intend to use it for
- whether you intend to use it only for research and will not include it in any commercial product OR you would like a commercial license



************************************************************
Terms of Use: 
************************************************************

1. Cite the papers associated with the lexicon in your research papers and articles that make use of it. (The papers associated with this lexicon are listed below.) 

2. In news articles and online posts on work using the lexicons, cite the appropriate lexicons. For example:
"This application/product/tool makes use of the <resource name>, created by <author(s)> at the National Research Council Canada." (The creators of this lexicon are listed below. Also, if you send us an email, we will be thrilled to know about how you have used the lexicon.) If possible hyperlink to this page: http://saifmohammad.com/WebPages/lexicons.html

3. If you use a lexicon in a product or application, then acknowledge this in the 'About' page and other relevant documentation of the application by stating the name of the resource, the authors, and NRC. For example:
"This application/product/tool makes use of the <resource name>, created by <author(s)> at the National Research Council Canada." (The creators of this lexicon are listed below. Also, if you send us an email, we will be thrilled to know about how you have used the lexicon.) If possible hyperlink to this page: http://saifmohammad.com/WebPages/lexicons.html

4. Do not redistribute the data. Direct interested parties to this page: http://saifmohammad.com/WebPages/AccessResource.htm

5. If interested in commercial use of this lexicon, see information above ("Accessing the NRC Sentiment and Emotion Lexicons").

6. National Research Council Canada (NRC) disclaims any responsibility for the use of the lexicon and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.




Creators: Saif M. Mohammad and Peter D. Turney

Papers associated with this lexicon:

Saif Mohammad and Peter Turney. Crowdsourcing a Word-Emotion Association Lexicon. Computational Intelligence, 29(3): 436-465, 2013. Wiley Blackwell Publishing Ltd.
 	 
Saif Mohammad and Peter Turney. Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon. In Proceedings of the NAACL-HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010, LA, California.





************************************************************
GENERAL DESCRIPTION
************************************************************

The NRC Emotion Lexicon is a list of words and their associations with eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). The annotations were manually done through Amazon's Mechanical Turk. 



************************************************************
FORMS OF THE LEXICON
************************************************************

1. Annotations at word-sense level (file: NRC-Emotion-Lexicon-Senselevel-v0.92.txt)

The original lexicon has annotations at word-sense level. Each word-sense pair was annotated by at least three annotators (most are annotated by at least five). 

2. Annotations at word level (file: NRC-Emotion-Lexicon-Wordlevel-v0.92.txt)

The word-level lexicon was created by taking the union of emotions associated with all the senses of a word. 

3. Translation into 40 languages (file: NRC-Emotion-Lexicon-v0.92-InManyLanguages.xlsx)

The NRC Emotion Lexicon has affect annotations for English words. Despite some cultural differences, it has been shown that a majority of affective norms are stable across languages. Thus, we provide versions of the lexicon in forty languages by translating the English terms using Google Translate (July 2015). We currently have translations into these languages: Arabic, Basque, Bengali, Catalan, Chinese, Chinese, Danish, Dutch, Esperanto, Finnish, French, German, Greek, Gujarati, Hebrew, Hindi, Italian, Irish, Japanese, Latin, Marathi, Persian, Portuguese, Romanian, Russian, Somali, Spanish, Sudanese, Swahili, Swedish, Tamil, Telugu, Thai, Turkish, Ukranian, Urdu, Vietnamese, Welsh, and Zulu. 

Note that some translations by Google Translate may be incorrect or they may simply be transliterations of the original English terms. 



************************************************************
FILE FORMAT 
************************************************************

Annotations at WORD-SENSE LEVEL (file: NRC-Emotion-Lexicon-Senselevel-v0.92.txt)

Each line has the following format:
<term>--<NearSynonyms><tab><AffectCategory><tab><AssociationFlag>

<term> is a word for which emotion associations are provided;

<NearSynonyms> is a set of one to three comma-separated words that indicate the sense of the <term>. The affect annotations are for this sense of the term.

<AffectCategory> is one of eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, or disgust) or one of two polarities (negative or positive);

<AssociationFlag> has one of two possible values: 0 or 1. 0 indicates that the target word has no association with affect category, whereas 1 indicates an association.



Annotations at WORD LEVEL (file: NRC-Emotion-Lexicon-Wordlevel-v0.92.txt)

Each line has the following format:
<term><tab><AffectCategory><tab><AssociationFlag>

<term> is a word for which emotion associations are provided;

<AffectCategory> is one of eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, or disgust) or one of two polarities (negative or positive);

<AssociationFlag> has one of two possible values: 0 or 1. 0 indicates that the target word has no association with affect category, whereas 1 indicates an association.



************************************************************
VERSION INFORMATION
************************************************************

Version 0.92 is the latest version as of 10 July 2011. This version has annotations for more than twice as many terms as in Version 0.5.



************************************************************
More Information
************************************************************

Details on the process of creating the lexicon can be found in the associated papers (see above).

 
