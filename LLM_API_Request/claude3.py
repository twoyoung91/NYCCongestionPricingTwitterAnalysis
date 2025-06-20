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


import anthropic

token = 'YOUR_ANTHROPIC_API_KEY'  # Replace with your actual API key

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=token,
)

message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)


sample_df=pd.read_csv("output/sample_all_231201.csv")

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
def ownertype_claude(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. Is this a personal account or an organizational account? Answer this question with personal, organizational, or not sure without explanation"
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    )
    return response

# #test case
# user = sample_df.loc[0,]
# response = ownertype_claude(user)
# answer=response.content[0].text
# print(answer)

ownertype_result=[]
for index, row in sample_df.iterrows():
    response = ownertype_claude(row)
    answer = response.content[0].text
    ownertype_result+=[answer]
    time.sleep(1)
    print(index)

sample_df["claude_otype"]=ownertype_result
sample_df["claude_otype"]=sample_df["claude_otype"].str.lower()
sample_df["OwnerType"]=sample_df["OwnerType"].str.lower()
sample_df["claude_otype"].unique()
sample_df["OwnerType"].unique()

accuracy_score(sample_df["OwnerType"], sample_df["claude_otype"])
precision_recall_fscore_support(sample_df["OwnerType"], sample_df["claude_otype"], average='weighted')


#Gender
personal_df=sample_df[sample_df['OwnerType']=='personal']
def gender_claude(row):
    accid = row["name"]
    username = row["username"]
    description = row["description"]
    prompt = f"This is the Twitter account name '{accid}', and its username is '{username}'. try your best to guess the most likely gender of the owner of this account. Answer gender with male or female. Answer not sure if you don't have any information to speculate. No explanation."
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    )
    return response

gender_result=[]

for index, row in personal_df.iterrows():
    response = gender_claude(row)
    answer = response.content[0].text
    gender_result+=[answer]
    print(index)
    
personal_df["claude_gender"]=gender_result
personal_df["claude_gender"]=personal_df["claude_gender"].str.lower()
personal_df["claude_gender"].unique()

accuracy_score(personal_df["gender"], personal_df["claude_gender"])
precision_recall_fscore_support(personal_df["gender"], personal_df["claude_gender"], average='weighted')


#Ethnicity
def ethnicity_claude(row):
    accid = row["name"]
    username = row["username"]
    description = row["description"]
    prompt = f"The account is named '{accid}', with the username '{username}'. What is its most possible ethnic origin? Answer with 'White', 'Hispanic or Latino', 'Black', 'Asian, Native Hawaiian and Other Pacific Islander', 'American Indian and Alaska Native'. Answer not sure if you don't have a name to process. No explanation."
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    )
    return response

ethnicity_result=[]

for index, row in personal_df.iterrows():
    response = ethnicity_claude(row)
    answer = response.content[0].text
    ethnicity_result+=[answer]
    time.sleep(1)
    print(index)
    
personal_df["claude_ethnicity"]=ethnicity_result
personal_df["claude_ethnicity"]=personal_df["claude_ethnicity"].str.lower()
personal_df.loc[personal_df["claude_ethnicity"]=="asian, native hawaiian and other pacific islander","claude_ethnicity"]="asian"
personal_df["claude_ethnicity"].unique()


accuracy_score(personal_df["ethnicity"], personal_df["claude_ethnicity"])
precision_recall_fscore_support(personal_df["ethnicity"], personal_df["claude_ethnicity"], average='weighted')

#Occupupation

def occupation_claude(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name \"{accid}\", and it is an \"{verified}\" account. Account description is \"{description}\". Does the owner introduce his job among those information? reply job name only. reply \"not sure\" if it is hard to determine. no explanation."
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    )  
    return response


def soc2018_claude(row):
    jobname = row["claude_occupation"]
    prompt = f"What is the class that job \"{jobname}\ belongs to in the U.S Standard Occupational Classification System 2018? Use the highest level classification with 23 categories in this system. Answer only the full name of the classification. No explanation."
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    )  
    return response

occupation_result=[]

for index, row in personal_df.iterrows():
    response = occupation_claude(row)
    answer = response.content[0].text
    occupation_result+=[answer]
    print(index)
    
personal_df["claude_occupation"]=occupation_result
personal_df["claude_occupation"]=personal_df["claude_occupation"].str.lower()
personal_df["claude_occupation"].unique()

soc_result=[]

for index, row in personal_df.iterrows():
    if row["claude_occupation"]=="not sure":
        soc_result+=["not sure"]
        print(index)
        continue
    response = soc2018_claude(row)
    answer = response.content[0].text
    soc_result+=[answer]
    print(index)

personal_df["claude_occ_group"]=soc_result
personal_df["claude_occ_group"].unique()

accuracy_score(personal_df["Occupation"], personal_df["claude_occ_group"])
precision_recall_fscore_support(personal_df["Occupation"], personal_df["claude_occ_group"], average='weighted')

#Organization Type
org_df=sample_df[sample_df["OwnerType"]=="organizational"].reset_index()

def orgtype_claude(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. What is the type of this orgnaization? Reply with one from Government,Insititution, Enterprise, Nonprofit, Media, Others,or not sure. no explanation."
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    ) 
    return response


orgtype_result=[]
for index, row in org_df.iterrows():
    response = orgtype_claude(row)
    answer = response.content[0].text
    orgtype_result+=[answer]
    print(index)
    
org_df["claude_org_type"]=orgtype_result
org_df["claude_org_type"].unique()
org_df["claude_org_type"]=org_df["gpt_org_type"].str.replace('"', '')

accuracy_score(org_df["OrganizationalType"], org_df["claude_org_type"])
precision_recall_fscore_support(org_df["OrganizationalType"], org_df["claude_org_type"], average='weighted')

#Sentiment
def analyze_sentiment_claude(tweet):
    prompt = (f"Is this tweet in support, opposition, or neutral on the NYC congestion plan: \"{tweet}\"? Reply with support, oppose, or neutral without explanation.")
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    ) 
    return response

sent_result=[]
for index, row in sample_df.iterrows():
    tweet = row["text"]
    response = analyze_sentiment_claude(tweet)
    answer = response.content[0].text
    sent_result+=[answer]
    print(index)
    
sample_df["claude_sent"]=sent_result
sample_df["claude_sent"]=sample_df["claude_sent"].str.lower()
sample_df["claude_sent"].unique()

conf_matrix_tweet_claude = confusion_matrix(sample_df["Opinion"],sample_df["claude_sent"],labels=["support", "oppose", "neutral"])    


#keywords
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def keyword_claude(tweet):
    prompt = (f"Extract keywords from this tweet: \"{tweet}\". Reply directly with a list of keywords, no explanation or starting word.")
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    ) 
    return response


keyword_result=[]
for index, row in sample_df.iterrows():
    tweet = row["text"]
    response = keyword_claude(tweet)
    answer = response.content[0].text
    keyword_result+=[answer]
    print(index)
    

sample_df['claude_keywords']=keyword_result
sample_df['claude_keywords']=sample_df['claude_keywords'].str.replace('\n', ', ', regex=False)

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
sample_df["claude_keywords"]=sample_df["claude_keywords"].apply(clean_words)
sample_df["claude_kw_match"]=sample_df.apply(compare_word_lists,args=("Keyword","claude_keywords"),axis=1)
sample_df["claude_kw_match"].sum()


sample_df_full=pd.merge(sample_df,personal_df[['id_x','claude_gender','claude_ethnicity','claude_occupation','claude_occ_group']],on='id_x',how='left')
sample_df_full=pd.merge(sample_df_full,org_df[['id_x','claude_org_type']],on='id_x',how='left')
sample_df_full.to_csv('output/sample_claude_240724.csv')


### sentiment 140 ###

sen140_df=pd.read_csv("output\sent140_sentiment_230913_1.csv",encoding='latin-1')

def analyze_sentiment_claude(tweet):
    prompt = (f"Please analyze the sentiment of the following tweet: \"{tweet}\". Reply with positive, negative, or neutral. No explanation.")
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    ) 
    return response

sent_result=[]
for index, row in sen140_df.iterrows():
    tweet = row["text"]
    response = analyze_sentiment_claude(tweet)
    answer = response.content[0].text
    sent_result+=[answer]
    print(index)
    
sen140_df["claude_sent"]=sent_result
sen140_df["claude_sent"]=sen140_df["claude_sent"].str.lower()
sen140_df["claude_sent"].unique() 

conf_matrix_tweet_claude = confusion_matrix(sen140_df["true_sent"],sen140_df["claude_sent"],labels=["positive", "negative", "neutral"])    
sen140_df.to_csv('output/sentiment140/sent140_claude_270724.csv')

### Inspec Keyword ###
inspec_df=pd.read_csv('output/inspec/inspec_gpt_230915_1.csv')

def keyword_claude(tweet):
    prompt = (f"Extract keywords from this tweet: \"{tweet}\". Reply directly with a list of keywords, no explanation or starting word.")
    messages =[{"role":"user","content":prompt}]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages= messages
    ) 
    return response


keyword_result=[]
for index, row in inspec_df.iterrows():
    tweet = row["text"]
    response = keyword_claude(tweet)
    answer = response.content[0].text
    keyword_result+=[answer]
    print(index)
    

inspec_df['claude_keywords']=keyword_result
inspec_df['claude_keywords']=inspec_df['claude_keywords'].str.replace('\n', ', ', regex=False)
inspec_df['Keyword']=inspec_df['Keyword'].apply(lambda x: ",".join(ast.literal_eval(x)))

inspec_df.to_csv('output\inspec\inspec_claude_270724.csv',index=False)

inspec_df["Keyword"]=inspec_df["Keyword"].apply(clean_words)
inspec_df["claude_keywords"]=inspec_df["claude_keywords"].apply(clean_words)
inspec_df["claude_kw_match"]=inspec_df.apply(compare_word_lists,args=("Keyword","claude_keywords"),axis=1)
inspec_df["claude_kw_match"].sum()
