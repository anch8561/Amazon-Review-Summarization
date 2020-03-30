#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk.data
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# In[2]:


review  = "It's a sample sentence with random words to test if each of the steps work!! Thinking, wondering , writing and watching wayyyyy tooooooo much. am, are, was, were, be, is, A, SFO, I, happpppyyyyy.... youuuu, By the way, I finished the last piece of cake and wrote the first draft without much of a / any trouble. BTW, don't do it, lol, was sleeping, gibberish, add the fairness measure we discussed, contact @ amazon, $50 value, 100%, jack & gill, *lol*, (surprised)"
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

tokens = tokenizer.tokenize(review)

list_of_words = []
for word in tokens:
    if word.isalnum() and not word in stop_words:
        word = word.lower()
        word = lemmatizer.lemmatize(word, pos = 'v')
        word = porter.stem(word)
        list_of_words.append(word)

print(list_of_words)


# In[ ]:




