# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:42:25 2024

@author: TwoYoung
"""

import os
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score,precision_recall_fscore_support
import matplotlib.pyplot as plt
#import seaborn as sns
import time
import ollama


sample_df=pd.read_csv("output/sample_llama3_240731.csv")

#remove same content tweets
def removeurl(tweet):
    wordlist=tweet.split()
    for xword in ['https','uhttps']:
        wordlist=list(word for word in wordlist if word.find(xword)==-1)  
    newtweet=" ".join(wordlist)
    return newtweet

sample_df['text']=sample_df['text'].astype('str').apply(removeurl)
sample_df.loc[sample_df["verified"]==True,"veriacc"]="verified"
sample_df.loc[sample_df["verified"]==False,"veriacc"]="unverified"

#speculate personal/organizational
def ownertype_qwen2(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. Is this a personal account or an organizational account? Only reply with one word： 'personal', 'organizational', or 'not sure'. You don't need to provide explanation"
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

# #test case
# user = sample_df.loc[0,]
# response = ownertype_qwen2(user)
# answer=response.content[0].text
# print(answer)

ownertype_result=[]
for index, row in sample_df.iterrows():
    response = ownertype_qwen2(row)
    answer = response['message']['content']
    ownertype_result+=[answer]
    print(index)

sample_df["qwen2_otype"]=ownertype_result
sample_df["qwen2_otype"]=sample_df["qwen2_otype"].str.lower()
sample_df["OwnerType"]=sample_df["OwnerType"].str.lower()
sample_df["qwen2_otype"].unique()
sample_df["OwnerType"].unique()

accuracy_score(sample_df["OwnerType"], sample_df["qwen2_otype"])
precision_recall_fscore_support(sample_df["OwnerType"], sample_df["qwen2_otype"], average='weighted')


#Gender
personal_df=sample_df[sample_df['OwnerType']=='personal']
def gender_qwen2(row):
    accid = row["name"]
    username = row["username"]
    description = row["description"]
    prompt = f"This is the Twitter account name '{accid}', and its username is '{username}'. try your best to guess the most likely gender of the owner of this account. Answer gender with male or female. Answer not sure if you don't have any information to speculate. No explanation."
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

gender_result=[]

for index, row in personal_df.iterrows():
    response = gender_qwen2(row)
    answer = response['message']['content']
    gender_result+=[answer]
    print(index)
    
personal_df["qwen2_gender"]=gender_result
personal_df["qwen2_gender"]=personal_df["qwen2_gender"].str.lower()
personal_df["qwen2_gender"].unique()

accuracy_score(personal_df["gender"], personal_df["qwen2_gender"])
precision_recall_fscore_support(personal_df["gender"], personal_df["qwen2_gender"], average='weighted')


#Ethnicity
def ethnicity_qwen2(row):
    accid = row["name"]
    username = row["username"]
    description = row["description"]
    prompt = f"The account is named '{accid}', with the username '{username}'. What is its most possible ethnic origin? Only reply with one word：'White', 'Hispanic or Latino', 'Black', 'Asian', 'American Indian and Alaska Native', or 'not sure'. No explanation."
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

ethnicity_result=[]

for index, row in personal_df.iterrows():
    response = ethnicity_qwen2(row)
    answer = response['message']['content']
    ethnicity_result+=[answer]
    print(index)
    
personal_df["qwen2_ethnicity"]=ethnicity_result
personal_df["qwen2_ethnicity"]=personal_df["qwen2_ethnicity"].str.lower()
personal_df.loc[personal_df["qwen2_ethnicity"]=="asian, native hawaiian and other pacific islander","qwen2_ethnicity"]="asian"
personal_df["qwen2_ethnicity"].unique()


accuracy_score(personal_df["ethnicity"], personal_df["qwen2_ethnicity"])
precision_recall_fscore_support(personal_df["ethnicity"], personal_df["qwen2_ethnicity"], average='weighted')

#Occupupation

def occupation_qwen2(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name \"{accid}\", and it is an \"{verified}\" account. Account description is \"{description}\". Does the owner introduce his job among those information? reply job name only. reply \"not sure\" if it is hard to determine. no explanation."
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

def soc2018_qwen2(row):
    jobname = row["qwen2_occupation"]
    prompt = f"What is the class that job \"{jobname}\ belongs to in the U.S Standard Occupational Classification System 2018? Use the highest level classification with 23 categories in this system. Answer only the full name of the classification. No explanation."
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

occupation_result=[]

for index, row in personal_df.iterrows():
    response = occupation_qwen2(row)
    answer = response['message']['content']
    occupation_result+=[answer]
    print(index)
    
personal_df["qwen2_occupation"]=occupation_result
personal_df["qwen2_occupation"]=personal_df["qwen2_occupation"].str.lower()
personal_df["qwen2_occupation"].unique()

soc_result=[]

for index, row in personal_df.iterrows():
    if row["qwen2_occupation"]=="not sure":
        soc_result+=["not sure"]
        print(index)
        continue
    response = soc2018_qwen2(row)
    answer = response['message']['content']
    soc_result+=[answer]
    print(index)

personal_df["qwen2_occ_group"]=soc_result
personal_df["qwen2_occ_group"].unique()

accuracy_score(personal_df["Occupation"], personal_df["qwen2_occ_group"])
precision_recall_fscore_support(personal_df["Occupation"], personal_df["qwen2_occ_group"], average='weighted')

#Organization Type
org_df=sample_df[sample_df["OwnerType"]=="organizational"].reset_index()

def orgtype_qwen2(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. What is the type of this orgnaization? Reply with one from Government,Insititution, Enterprise, Nonprofit, Media, Others,or not sure. no explanation."
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response


orgtype_result=[]
for index, row in org_df.iterrows():
    response = orgtype_qwen2(row)
    answer = response['message']['content']
    orgtype_result+=[answer]
    print(index)
    
org_df["qwen2_org_type"]=orgtype_result
org_df["qwen2_org_type"].unique()
org_df["qwen2_org_type"]=org_df["gpt_org_type"].str.replace('"', '')

accuracy_score(org_df["OrganizationalType"], org_df["qwen2_org_type"])
precision_recall_fscore_support(org_df["OrganizationalType"], org_df["qwen2_org_type"], average='weighted')

#Sentiment
def analyze_sentiment_qwen2(tweet):
    prompt = (f"Is this tweet in support, opposition, or neutral on the NYC congestion plan: \"{tweet}\"? Reply with support, oppose, or neutral without explanation.")
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

sent_result=[]
for index, row in sample_df.iterrows():
    tweet = row["text"]
    response = analyze_sentiment_qwen2(tweet)
    answer = response['message']['content']
    sent_result+=[answer]
    print(index)
    
sample_df["qwen2_sent"]=sent_result
sample_df["qwen2_sent"]=sample_df["qwen2_sent"].str.lower()
sample_df["qwen2_sent"].unique()

conf_matrix_tweet_qwen2 = confusion_matrix(sample_df["Opinion"],sample_df["qwen2_sent"],labels=["support", "oppose", "neutral"])    


#keywords
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def keyword_qwen2(tweet):
    prompt = (f"Extract keywords from this tweet: \"{tweet}\". Reply directly with a list of no more than five keywords, no explanation or starting word. Separate keywords by comma.")
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response


keyword_result=[]
for index, row in sample_df.iterrows():
    tweet = row["text"]
    response = keyword_qwen2(tweet)
    answer = response['message']['content']
    keyword_result+=[answer]
    print(index)
    

sample_df['qwen2_keywords']=keyword_result
sample_df['qwen2_keywords']=sample_df['qwen2_keywords'].str.replace('\n', ', ', regex=False)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def towordlist(text):
    newtext=re.split(",| ",str(text))
    newtext = [word.lower() for word in newtext if word!=""]
    return newtext

def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Custom function to singularize verbs and nouns in a list of words
def clean_words(word):
    # POS tagging
    word_list=towordlist(word)
    
    pos_tagged = nltk.pos_tag(word_list)
    
    # Lemmatize using POS tag
    singular_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
    
    return singular_words


def compare_word_lists(row,column1,column2):
    list_a=row[column1]
    list_b=row[column2]
    # Using set intersection to find common elements
    if len(set(list_a))==0:
        return 1
    
    common_elements = set(list_a).intersection(set(list_b))
    IoU=len(common_elements)/len(set(list_a))
    
    # Return 1 if there are common elements, otherwise 0
    return 1 if IoU>0.5 else 0


sample_df["Keyword"]=sample_df["Keyword"].apply(clean_words)
sample_df["qwen2_keywords"]=sample_df["qwen2_keywords"].apply(clean_words)
sample_df["qwen2_kw_match"]=sample_df.apply(compare_word_lists,args=("Keyword","qwen2_keywords"),axis=1)
sample_df["qwen2_kw_match"].sum()


sample_df_full=pd.merge(sample_df,personal_df[['id_x','qwen2_gender','qwen2_ethnicity','qwen2_occupation','qwen2_occ_group']],on='id_x',how='left')
sample_df_full=pd.merge(sample_df_full,org_df[['id_x','qwen2_org_type']],on='id_x',how='left')
sample_df_full.to_csv('output/sample_qwen2_240724.csv')

#sentiment 140

sen140_df=pd.read_csv("output\sentiment140\sent140_claude_270724.csv",encoding='latin-1')

def analyze_sentiment_qwen2(tweet):
    prompt = (f"Please analyze the sentiment of the following tweet: \"{tweet}\". Reply with 'positive', 'negative', or 'neutral'. No explanation.")
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response

sent_result=[]
for index, row in sen140_df.iterrows():
    tweet = row["text"]
    response = analyze_sentiment_qwen2(tweet)
    answer = response['message']['content']
    sent_result+=[answer]
    print(index)
    
sen140_df["qwen2_sent"]=sent_result
sen140_df["qwen2_sent"]=sen140_df["qwen2_sent"].str.lower()
sen140_df["qwen2_sent"]=sen140_df["qwen2_sent"].str.replace('mixed','neutral')
sen140_df["qwen2_sent"].unique() 

conf_matrix_tweet_qwen2 = confusion_matrix(sen140_df["true_sent"],sen140_df["qwen2_sent"],labels=["positive", "negative", "neutral"])    
sen140_df.to_csv('output/sentiment140/sent140_qwen2_270724.csv')


### Inspec Keyword ###
inspec_df=pd.read_csv('output/inspec/inspec_claude_270724.csv')

def keyword_qwen2(tweet):
    prompt = (f"Extract keywords from this tweet: \"{tweet}\". Reply directly with a list of keywords, no explanation or starting word.")
    messages =[{"role":"user","content":prompt}]
    response=ollama.chat(model='qwen2', messages=messages)
    return response


keyword_result=[]
for index, row in inspec_df.iterrows():
    tweet = row["text"]
    response = keyword_qwen2(tweet)
    answer = response['message']['content']
    keyword_result+=[answer]
    print(index)
    

inspec_df['qwen2_keywords']=keyword_result

inspec_df.to_csv('output\inspec\inspec_qwen2_270724.csv',index=False)

inspec_df["Keyword"]=inspec_df["Keyword"].apply(clean_words)
inspec_df["qwen2_keywords"]=inspec_df["qwen2_keywords"].apply(clean_words)
inspec_df["qwen2_kw_match"]=inspec_df.apply(compare_word_lists,args=("Keyword","qwen2_keywords"),axis=1)
inspec_df["qwen2_kw_match"].sum()
