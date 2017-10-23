NRC Word-Colour Association Lexicon (NRC Colour Lexicon)
Version 0.92
21 July 2011
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




Creator: Saif M. Mohammad

Papers associated with this lexicon:

Saif M. Mohammad. Colourful Language: Measuring Word-Colour Associations. In Proceedings of the ACL 2011 Workshop on Cognitive Modeling and Computational Linguistics (CMCL), June 2011, Portland, OR.

Saif M. Mohammad. Even the Abstract have Colour: Consensus in Word-Colour Associations. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, June 2011, Portland, OR.



************************************************************
GENERAL DESCRIPTION
************************************************************

Many real-world concepts have associations with colours. For example, iceberg is associated with white, vegetation with green, danger with red, and so on. The NRC word-colour association lexicon is a list of
words and the colours they are most associated with. The annotations were manually done through Amazon's Mechanical Turk. 



************************************************************
FILE FORMAT
************************************************************

Each line has the following format:
<TargetWord>--<sense><tab>Colour=<colour><tab>VotesForThisColour=<VotesForThisColour><tab>TotalVotesCast=<TotalVotesCast>

<TargetWord> is a word for which the annotators provided colour associations;

<sense> is one or more comma-separated words that indicate the sense of the target word for which the annotations are provided;

<colour> is the colour most associated with the target word. It is one of eleven colours---white, black, red, green, yellow, blue, brown, pink, purple, orange, grey. If each of the annotators suggested a different colour association for the target word, then <colour> is set to None.

<VotesForThisColour> is the number of annotators who chose <colour> for the target word. It is set to None if <colour> is None.

<TotalVotesCast> is the total number of annotators who gave colour associations for the target word.


************************************************************
VERSION INFORMATION
************************************************************

Version 0.92 is the latest version as of 21 July 2011. 


************************************************************
More Information
************************************************************

Details on the process of creating the lexicon can be found in the associated papers (see above).
 

