import re
import nltk
import requests
import pandas as pd

nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def clean_tweet(tweet):
    # removal of @name[mention]
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    # removal of links[https://blabala.com]
    tweet = re.sub(r"http\S+", "", tweet)
    # removal of new line
    tweet = re.sub('\n', '', tweet)
    # removal of RT
    tweet = re.sub('RT', '', tweet)
    # removal of punctuations and numbers
    tweet = re.sub("[^a-zA-Z^']", " ", tweet)
    tweet = re.sub(" {2,}", " ", tweet)
    # remove leading and trailing whitespace
    tweet = tweet.strip()
    # remove whitespace with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # convert text to Lowercase
    tweet = tweet.lower();
    return tweet

def tokenized(text):
    return nltk.word_tokenize(text)

def stemming(data):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return [stemmer.stem(tweet) for tweet in data]

# Slangword
slang_word = requests.get('https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/master/combined_slang_words.txt').text
dict_slang = eval(slang_word)
slang_df = pd.DataFrame(dict_slang.items(), columns=['Old', 'New'])
slang_df['Old'] = slang_df['Old'].apply(lambda x: x.strip())
slang_df['New'] = slang_df['New'].apply(lambda x: x.strip())
slang_dict = {}
for i, row in slang_df.iterrows():
  slang_dict.update({row['Old']: row['New']})
  
def removeSlang(data):
    for i, word in enumerate(data):
        try:
            data[i] = slang_dict[word]
        except KeyError:
            pass
    return data

# Stopword
nltk.download('stopwords')
from nltk.corpus import stopwords

def removeStopWords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [token for token in tokens if token not in stop_words]

def preProcessText(tw):
    tw = clean_tweet(tw)
    tw = tokenized(tw)
    tw = stemming(tw)
    tw = removeSlang(tw)
    tw = removeStopWords(tw)
    tw = ' '.join(tw)
    return tw