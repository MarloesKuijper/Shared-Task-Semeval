NRC Hashtag Emotion Lexicon
Version 0.2
4 November 2013
Copyright (C) 2013 National Research Council Canada (NRC)
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




Creator: Saif M. Mohammad

Papers associated with this lexicon:

Saif M. Mohammad and Svetlana Kiritchenko. Using Hashtags to Capture Fine Emotion Categories from Tweets. 
Computational Intelligence, 31(2): 301-326, 2015.

Saif M. Mohammad. #Emotional Tweets. In Proceedings of the First Joint Conference on Lexical and Computational Semantics (*Sem), June 2012, Montreal, Canada. 




************************************************************
GENERAL DESCRIPTION
************************************************************

The NRC Hashtag Emotion Lexicon is a list of words and their associations with eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust). The associations were computed from tweets with emotion-word hashtags such as #happy and #anger. 

The NRC Hashtag Emotion Lexicon is different from the NRC Emotion Lexicon. The Hashtag lexicon was automatically generated from tweets using emotion word hashtags. The NRC Emotion Lexicon was created by crowdsourcing and direct human annotation. 


************************************************************
DATA SOURCE
************************************************************

The NRC Hashtag Emotion Lexicon is automatically generated from the following data source: 
tweets with emotion-word hashtags collected by NRC.


************************************************************
FILE FORMAT
************************************************************

Each line has the following format:
<AffectCategory><tab><term><tab><score>

<AffectCategory> is one of eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, or disgust).

<term> is a word for which emotion associations are provided.

<score> is a value that indicates the strength of association between the <term> and the <AffectCategory>. The higher the value, the stronger is the association. Refer to the publications below for details on how the score is calculated.


************************************************************
VERSION INFORMATION
************************************************************

Version 0.2 is the latest version as of 4 November 2013. Version 0.1 was created using eight hashtags (pertaining to the eight basic emotions). Version 0.2 was created using 72 hashtags (these include the eight used in version 0.1 as well as their synonyms.) The publications mentioned below describe the creation and use of version 0.1. The use of version 0.2 in the same experiments further improves results.


************************************************************
More Information
************************************************************

Details on the process of creating the lexicon can be found in the associated papers (see above).

 
