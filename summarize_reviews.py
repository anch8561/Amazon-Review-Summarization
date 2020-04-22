from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize

def summarize_reviews(topics, reviews):
    # topics: list of strings. Each string contains the topics for a review
    # reviews: list of dicts with the following fields
    #  'reviewText': string with text of the review
    #  'topic': topics index
    # returns reviews with the following new fields
    #  'summary': sentences from review w/ topic words

    analyser = SentimentIntensityAnalyzer()

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

    return reviews


## TESTING

reviews = [{'topic': 0, 'reviewText': 'Wow, I love this product so much. It is the greatest. The RGB is so fun! I wish it were cheaper, $2M is a lot for a toaster. Plus it burns my toast. TLDR: Cool RGB but too expensive'}]
topics = ['rgb toast expensive']
reviews = summarize_reviews(topics, reviews)

print('Review: ' + reviews[0]['reviewText'])
print('Topics: ' + topics[0])
print('Summary: ' + reviews[0]['summary'])
    
## look for keywords like 'TLDR', 'summary', 'conclusion'