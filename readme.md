# Shared Task Semeval 2017/2018

Detecting affect intensity in tweets.
# To Do:

  - Have both text and emotion features
  - Add more features/affect lexicon integrations
  - Test various classifiers and different parameter settings
  - Cross validation to prevent overfitting and test quality of models > keep this consistent        when testing different models, so we can compare them
  - Incorporate WEKA system ??? Java system though

### Purpose of this readme

Use this readme to show which models you have used with which parameters/features. Like this:

Name - Date - Short description of what you did
```sh
For the pipeline:
TfidfVectorizer with settings: min_df=50, etc. etc.
DictVectorizer with settings bla and bla
Results: Pearson R = X

Or alternatively:
New model:
insert model
Results: Pearson R = X
```
AGAIN: It is crucial that we keep testing the R coefficient of all these different models in the same way. Otherwise we will not be able to compare the models / different parameter settings to each other.

### Contributors


| Author | 
| ------ |
| Mike van Lenthe | 
| Daniel van der Hall | 
| Marloes Kuijper| 