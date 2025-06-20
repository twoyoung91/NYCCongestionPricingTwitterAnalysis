# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:03:48 2023

@author: TwoYoung
"""


import openai_secret_manager
import openai
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re

openai.api_key = "enter_your_api_key_here"  # Replace with your actual OpenAI API key

filelist=os.listdir(r"input\inspec\docsutf8")
inspec_df=pd.DataFrame(columns=["text","Keyword"])

for file in filelist:
    docpath=os.path.join(r"input\inspec\docsutf8",file)
    keypath=os.path.join(r"input\inspec\keys",file[:-4]+".key")
    with open(docpath, 'r') as f:
        doc = f.read()
    doc=doc.replace('\t', '')
    doc=doc.replace('\n', ' ')
    with open(keypath, 'r') as f:
        keys = f.readlines() 
    key=[key.strip().replace('\t', '') for key in keys]
    inspec_df.loc[inspec_df.shape[0],]=[doc,key]

sample_df=inspec_df.sample(n=400,random_state=1991).reset_index(drop=True)

#remove url
def removeurlandat(tweet):
    wordlist=tweet.split()
    for xword in ['#','@','https','uhttps']:
        wordlist=list(word for word in wordlist if word.find(xword)==-1)  
    newtweet=" ".join(wordlist)
    return newtweet

sample_df['text']=sample_df['text'].apply(removeurlandat)

#GPT keywords

def keyword_gpt(tweet):
    prompt = (f"Extract keywords from this abstract: \"{tweet}\". Only reply the keywords. no more than five words, no explanation.")
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        timeout=15,
    )  
    return response


tweet = sample_df.loc[1,"text"]
response = keyword_gpt(tweet)
answer = response['choices'][0]['message']['content']
print(answer)


keyword_result=[]
for index, row in sample_df.iterrows():
    tweet = row["text"]
    response = keyword_gpt(tweet)
    answer = response['choices'][0]['message']['content']
    keyword_result+=[answer]
    print(index)

sample_df['gpt_keywords']=keyword_result
sample_df.to_csv(r"output\inspec_gpt_230915_1.csv")
keyword_gpt_df=sample_df[["Keyword","gpt_keywords"]]

#change keyword to word lists
def towordlist(text):
    newtext=re.split(",| ",str(text))
    newtext = [word.lower() for word in newtext if word!=""]
    return newtext

keyword_gpt_df["gpt_keywords"]=keyword_gpt_df["gpt_keywords"].apply(towordlist)
keyword_gpt_df["Keyword"]=keyword_gpt_df["Keyword"].apply(lambda x:[item for word in x for item in word.split(" ")])


# Function to map POS tag to first character used by WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Custom function to singularize verbs and nouns in a list of words
def singularize_words(word_list):
    # POS tagging
    pos_tagged = nltk.pos_tag(word_list)
    
    # Lemmatize using POS tag
    singular_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
    
    return singular_words

keyword_gpt_df["Keyword"]=keyword_gpt_df["Keyword"].apply(singularize_words)
keyword_gpt_df["gpt_keywords"]=keyword_gpt_df["gpt_keywords"].apply(singularize_words)


def compare_word_lists(row,column1,column2):
    list_a=row[column1]
    list_b=row[column2]
    # Using set intersection to find common elements
    common_elements = set(list_a).intersection(set(list_b))
    IoU=len(common_elements)/len(set(list_a))
    
    # Return 1 if there are common elements, otherwise 0
    return 1 if IoU>0.2 else 0

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def filter_stop_words(wordlist):
    newlist=[word.lower() for word in wordlist if not word in stop_words]
    return newlist

keyword_gpt_df["Keyword"]=keyword_gpt_df["Keyword"].apply(filter_stop_words)
keyword_gpt_df["gpt_keywords"]=keyword_gpt_df["gpt_keywords"].apply(filter_stop_words)


keyword_gpt_df["match"]=keyword_gpt_df.apply(compare_word_lists,args=("Keyword","gpt_keywords"),axis=1)
keyword_gpt_df["match"].sum()


#RAKE Keywords
from rake_nltk import Rake
r = Rake()
def rake_keyword(text):
    r.extract_keywords_from_text(text)
    wordlist=[x.split(" ") for x in r.get_ranked_phrases()[0:5]]
    combined_list = [item for sublist in wordlist for item in sublist]
    return  combined_list

sample_df['rake_keywords']=sample_df["text"].apply(rake_keyword)
keyword_df=sample_df[["Keyword","rake_keywords"]]
keyword_df["Keyword"]=keyword_df["Keyword"].apply(lambda x:[item for word in x for item in word.split(" ")])
keyword_df["Keyword"]=keyword_df["Keyword"].apply(singularize_words)
keyword_df["rake_keywords"]=keyword_df["rake_keywords"].apply(singularize_words)

keyword_df["Keyword"]=keyword_df["Keyword"].apply(filter_stop_words)
keyword_df["rake_keywords"]=keyword_df["rake_keywords"].apply(filter_stop_words)
keyword_df["match"]=keyword_df.apply(compare_word_lists,args=("Keyword","rake_keywords"),axis=1)
keyword_df["match"].sum()


