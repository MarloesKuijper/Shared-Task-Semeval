NRC Emoticon Affirmative Context Lexicon and NRC Emoticon Negated Context Lexicon
Version 1.0
26 September 2014
Copyright (C) 2014 National Research Council Canada (NRC)
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




Creators: Svetlana Kiritchenko and Saif M. Mohammad


Papers associated with this lexicon:

Kiritchenko, S., Zhu, X., Mohammad, S. (2014). Sentiment Analysis of Short Informal Texts. Journal of Artificial Intelligence Research, 50:723-762, 2014.

Mohammad, S., Kiritchenko, S., Zhu, X. (2013). NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets. In Proceedings of the seventh International Workshop on Semantic Evaluation Exercises (SemEval-2013), June 2013, Atlanta, USA. 

Zhu, X., Kiritchenko, S., Mohammad, S. (2014). NRC-Canada-2014: Recent Improvements in Sentiment Analysis of Tweets. In Proceedings of the eigth international workshop on Semantic Evaluation Exercises (SemEval-2014), August 2014, Dublin, Ireland.    



****************************************************
DATA SOURCE
****************************************************

The NRC Emoticon Lexicons are automatically generated from the following data source:
1.6 million tweets with emoticons collected by Go and colleagues (see Go, A., Bhayani, R., & Huang, L. Twitter sentiment classication using distant supervision. Tech. rep., Stanford University, 2009.)


****************************************************
FILE FORMAT
****************************************************

Each line in the lexicons has the following format:
<term><tab><score><tab><Npos><tab><Nneg>

<term> can be a unigram or a bigram;
<score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
<Npos> is the number of times the term appears in the positive class, ie. in tweets with positive emoticons;
<Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative emoticons.


****************************************************
NRC Emoticon Affirmative Context Lexicon (AffLex) 
and 
NRC Emoticon Negated Context Lexicon (NegLex)
****************************************************

Both parts, word-sentiment associations in affirmative contexts (AffLex) and word-sentiment associations in negated contexts (NegLex), of each lexicon are contained in the same file. The NegLex entries have suffixes '_NEG' or '_NEGFIRST'.

In the unigram lexicon:
'_NEGFIRST' is attached to terms that directly follow a negator;
'_NEG' is attached to all other terms in negated contexts (not directly following a negator).

In the bigram lexicon:
'_NEG' is attached to all terms in negated contexts.

Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached. 


****************************************************
More Information
****************************************************

Details on the process of creating the lexicon can be found in the associated papers (see above).
 
