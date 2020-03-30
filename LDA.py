import pandas as pd
import gzip
import nltk
import wordcloud
# nltk.download("stopwords")
import simplejson as json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora
from gensim import models
import operator
from wordcloud import WordCloud

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

df = getDF('Video_Games_5.json.gz')
# # df = getDF('Electronics_5.json.gz')
# print(df.head(1))
# print('----')
# df.columns
# # print(df['asin'].nunique())
df = df[df['asin'] == 'B00178630A']
# # print(df.asin.value_counts())
# print(df.iloc[0])
# print(df.shape)
# print(df.reviewText)

#Regular expression tokenizer
tokenizer = RegexpTokenizer(r'\w+')
doc_1 = df['reviewText'].iloc[0]
# Using one review
tokens = tokenizer.tokenize(doc_1.lower())
print('{} characters in string vs {} words in a list'.format(len(doc_1),len(tokens)))
print(tokens[:10])
nltk_stpwd = stopwords.words('english')
print(len(set(nltk_stpwd)))
print(nltk_stpwd[:10])
stopped_tokens = [token for token in tokens if not token in nltk_stpwd]
print(stopped_tokens[:10])

#Step 3
sb_stemmer = SnowballStemmer('english')
stemmed_tokens = [sb_stemmer.stem(token) for token in stopped_tokens]
print('---------------')
print(stemmed_tokens)
print('---------------')
print(df.iloc[0])
##
num_reviews = df.shape[0]
# num_reviews = 100
doc_set = [df['reviewText'].iloc[i] for i in range(num_reviews)]
texts = []
for doc in doc_set:
    tokens = tokenizer.tokenize(doc.lower())
    stopped_tokens = [token for token in tokens if not token in nltk_stpwd]
    stemmed_tokens = [sb_stemmer.stem(token) for token in stopped_tokens]
    texts.append(stemmed_tokens)  # Adds tokens to new list "texts"
print('3.1')
print(texts[1])

#Step 4
texts_dict = corpora.Dictionary(texts)
texts_dict.save('elec_review.dict')
print(texts_dict)
#Assess the mapping between words and their ids we use the token2id #method:
print("IDs 1 through 10: {}".format(sorted(texts_dict.token2id.items(), key=operator.itemgetter(1), reverse = False)[:10]))
#Here we assess how many reviews have word complaint in it
complaints = df.reviewText.str.contains("complaint").value_counts()
ax = complaints.plot.bar(rot=0)
"""
Attempting to see what happens if we ignore tokens that appear in less
than 30 documents or more than 20% documents.
"""
# texts_dict.filter_extremes(no_below=20, no_above=0.10)-----------------------------------------------------------------
print(sorted(texts_dict.token2id.items(), key=operator.itemgetter(1), reverse = False)[:10])


# Step 5: Converting the dictionary to bag of words calling it corpus here
corpus = [texts_dict.doc2bow(text) for text in texts]
print(len(corpus))
print(corpus)
print('*****')
#Save a corpus to disk in the sparse coordinate Matrix Market format in a serialized format instead of random
corpora.MmCorpus.serialize('amzn_elec_review.mm', corpus)


# Step 6: Fit LDA model
lda_model = models.LdaModel(corpus, alpha='auto', num_topics=5, id2word=texts_dict, passes=20)
# Choosing the number of topics based on various categories of electronics on Amazon
topics_list = lda_model.show_topics(num_topics=5, num_words=5)
# sorted_topics_list = list(sorted(topics_list, key=lambda x: x[1]))
# print(topics_list)

# long_string = ','.join(list(papers['paper_text_processed'].values))
# # Create a WordCloud object
# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# # Generate a word cloud
# wordcloud.generate(long_string)
# # Visualize the word cloud
# wordcloud.to_image()



# raw_query = 'game hard'
# query_words = raw_query.split()
# query = []
# for word in query_words:
#     # ad-hoc reuse steps from above
#     q_tokens = tokenizer.tokenize(word.lower())
#     q_stopped_tokens = [word for word in q_tokens if not word in nltk_stpwd]
#     q_stemmed_tokens = [sb_stemmer.stem(word) for word in q_stopped_tokens]
#     query.append(q_stemmed_tokens[0])
#
# print(query)
# # Words in query will be converted to ids and frequencies
# id2word = corpora.Dictionary()
# _ = id2word.merge_with(texts_dict)  # garbage
# # Convert this document into (word, frequency) pairs
# query = id2word.doc2bow(query)
# print(query)
# # Create a sorted list
# print(lda_model[query])

# sorted_list = list(sorted(lda_model[query], key=lambda x: x[1]))
# sorted_list
# print(sorted_list)
# # Assessing least related topics
# print("least")
# print(lda_model.print_topic(sorted_list[0][0]))  # least related
# # Assessing most related topics
# print("most")
# print(lda_model.print_topic(sorted_list[-1][0]) ) # most related
#
# lda_model.print_topics(-1)

