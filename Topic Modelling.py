#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF,  LatentDirichletAllocation
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from nltk import sent_tokenize
from nltk.corpus import stopwords

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas as pd
import numpy as np

import string
import spacy
import gzip
import simplejson as json
import nltk
import en_core_web_sm
nlp = en_core_web_sm.load()

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# In[2]:


#Script for SAVING DATA. Uncomment it when needed

"""import os
import gzip
import json

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

fileName = 'Electronics_5'
inputFileName  = fileName + '.json.gz'
if not os.path.exists(fileName): os.makedirs(fileName)

count = 0
ii = 0
data = {}
asin = '0'
for review in parse(inputFileName):
    if review['asin'] != asin:
        outputFileName = fileName + r'/' + asin + '.json'
        outputFile = open(outputFileName, 'w', newline='')
        json.dump(data, outputFile)
        outputFile.close
        ii = 0
        data = {}
        asin = review['asin']
    try: helpfulness = review['vote']
    except: helpfulness = '0'
    try:
        data[ii] = {
            'helpfulness': helpfulness,
            'rating': review['overall'],
            'text': review['reviewText']}
        ii += 1
    except: pass
    count += 1
    if count % 1e5 == 0: print(count)"""


# In[3]:


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


class CleanTextTransformer(TransformerMixin):
   
    def transform(self, X, **transform_params):
        #return [cleanText(text) for text in X]
        return [text for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
"""def cleanText(text):
    "this function removes new lines."
    text = text.strip().replace("\n", " ").replace("\r", " ")
    return text
"""

def tokenizeText(sample):
    "This function tokenizes text and does other preprocessing steps like Lemmatization and Stemming."

    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    
    #tokenize
    tokens = tokenizer.tokenize(sample)
    # lemmatize
    lemmas = []
    for word in tokens:
        if word.isalnum() and not word in stop_words:
            word = word.lower()
            word = lemmatizer.lemmatize(word, pos = 'v')
            lemmas.append(word)
    tokens = lemmas
    # white space removal and new line removal
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens

#------------------------------------------------------------------------------
def return_topics(vectorizer, clf, W, df, n_top_words, n_top_documents):
    print('return topics')
    topics, reviews = [], []
    features = vectorizer.get_feature_names()
    sentiment_analyser = SentimentIntensityAnalyzer()

    for topic_id, topic in enumerate(clf.components_):

        # grab the list of words describing the topic
        topic_word_list = []
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            topic_word_list.append(features[i])

        # split words in case there are some bigrams
        split_topic_word_list = []
        for word in topic_word_list:
            for splitted in word.split():
                split_topic_word_list.append(splitted)
        topic_words = list(set(split_topic_word_list))

        # append topic words as a single string
        topics.append(' '.join([word for word in topic_words]))

        # iterate for reviews for each topic
        topic_doc_indices = np.argsort(W[:, topic_id])[::-1][0:n_top_documents]

        for doc_ind in topic_doc_indices:
            review = df['reviewText'].iloc[doc_ind]

            # check if the review contains any of the topic words
            if any(word in review.lower() for word in topic_words):
                # analyse sentiment
                vader = sentiment_analyser.polarity_scores(review)
                # form the review - topic_id and sentiment data structure
                reviews.append(df.iloc[doc_ind].to_dict())
                reviews[-1]['topic'] = topic_id
                reviews[-1]['sentiment'] = vader['compound']

    return topics, reviews
#------------------------------------------------------------------------------

def summarize_reviews(topics, reviews):
    # topics: list of strings. Each string contains the topics for a review
    # reviews: list of dicts with the following fields
    #  'reviewText': string with text of the review
    #  'topic': topics index
    # returns reviews with the following new fields
    #  'summary': sentences from review w/ topic words

    analyser = SentimentIntensityAnalyzer()
    summary_all_review = []
    for ii, review in enumerate(reviews):
        summary = []
        sentences = sent_tokenize(review['reviewText'])
        topic_words = topics[review['topic']].split()

        for sentence in sentences:
            for word in topic_words:
                if word in sentence.lower():
                    summary.append(sentence)
                    break

        reviews[ii]['summary'] = ' '.join([sentence for sentence in summary])
        vader = analyser.polarity_scores(reviews[ii]['summary'])
        reviews[ii]['summary_sentiment'] = vader['compound']
        
        summary_all_review.append(reviews[ii]['summary'])

    return summary_all_review

def print_topics(test_asin):

    test_df = reviews_df[reviews_df['asin'] == test_asin].dropna()
    n_features, n_top_words, n_topics, n_top_documents = 1000, 3, 6, 3

    vectorizer = TfidfVectorizer(max_features=n_features,
                                 tokenizer=tokenizeText,
                                 stop_words='english',
                                 ngram_range=(1,2))

    clf = NMF(n_components=n_topics, random_state=1, solver='mu', beta_loss='frobenius')
   
    #clf = LatentDirichletAllocation(n_components = 5, max_iter = 5, learning_method ='online',learning_offset = 50.,random_state = 0)

    pipe = Pipeline([('cleanText', CleanTextTransformer()),('vectorizer', vectorizer), ('nmf', clf)])

    pipe.fit(test_df['reviewText'])
    transform = pipe.fit_transform(test_df['reviewText'])
    
    #topic identification
    topics, reviews = return_topics(vectorizer, clf, transform, test_df, n_top_words, n_top_documents)
    # review summarization
    summary = summarize_reviews(topics, reviews)
    print("Summary :\n", summary)
    print("Topics:")
    
    return topics, reviews


# In[4]:


#reviews_df = getDF('Video_Games_5.json.gz')
reviews_df = getDF('Electronics_5.json.gz')
print(reviews_df.head(4))


# In[5]:


topic, review = print_topics('B0000A576B')

print(topic)
print("\n Reviews: \n", review)


# In[6]:


topic, review = print_topics('B0000AZJZB')


print(topic)
print("\n Reviews: \n", review)


# In[ ]:





# In[ ]:




