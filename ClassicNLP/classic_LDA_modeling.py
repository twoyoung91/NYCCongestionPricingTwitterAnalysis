# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:34:12 2023

@author: TwoYoung
"""


import os
import pandas as pd
import re


df_raw=pd.read_csv(r"output\full_all_230916_2.csv")
df_raw=df_raw.drop(["Unnamed: 0"],axis=1)
df_raw=df_raw.drop(["Unnamed: 0.1"],axis=1)


import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens=lda_tokens
        elif token.orth_.startswith('@'):
            lda_tokens=lda_tokens
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
repeat_words=['congestion', 'pricing','congestionpricing']

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [token for token in tokens if token not in repeat_words]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

token_tweets=df_raw['text'].drop_duplicates().astype(str)
token_tweets=token_tweets.apply(prepare_text_for_lda)

from gensim import corpora
dictionary = corpora.Dictionary(token_tweets)
corpus = [dictionary.doc2bow(text) for text in token_tweets]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)

import pyLDAvis.gensim
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
