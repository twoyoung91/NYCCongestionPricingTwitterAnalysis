# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:16:19 2023

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


sample_df=pd.read_csv("result.csv")

# Set up the OpenAI API client
openai.api_key = "enter_your_api_key_here"  # Replace with your actual OpenAI API key


def get_answer(response):
    answer = response['choices'][0]['message']['content']
    return answer


#remove url
def removeurl(tweet):
    wordlist=tweet.split()
    for xword in ['https','uhttps']:
        wordlist=list(word for word in wordlist if word.find(xword)==-1)  
    newtweet=" ".join(wordlist)
    return newtweet

sample_df['text']=sample_df['text'].apply(removeurl)
sent_df=sample_df.drop_duplicates("text")

def analyze_sentiment_gpt4(tweet):
    prompt = (f"Is this tweet in support, opposition, or neutral on the NYC congestion plan: \"{tweet}\"? Reply with support, oppose, or neutral without explanation.")
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
    )  
    return response

sent_result=[]
for index, row in sent_df.iterrows():
    tweet = row["text"]
    response = analyze_sentiment_gpt4(tweet)
    answer = response['choices'][0]['message']['content']
    sent_result+=[answer]
    time.sleep(1)
    print(index)
    
sent_df["gpt_sent_response"]=sent_result
sent_df["gpt_sent_response"]=sent_df["gpt_sent_response"].str.lower()

df_withsent=pd.merge(sample_df,sent_df[["text","gpt_sent_response"]],on="text",how="left")
df_withsent.to_csv(r"output\full_withsent_230916_1.csv",index=False)

#SES analysis
ses_df=sample_df[["author_id","name","verified","description"]]
ses_df=ses_df.drop_duplicates("author_id").reset_index()
ses_df.loc[ses_df["verified"]==True,"veriacc"]="verified"
ses_df.loc[ses_df["verified"]==False,"veriacc"]="unverified"

def ownertype_gpt4(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. Is this a personal account or an organizational account? Answer this question with personal, organizational, or not sure without explanation"
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
    )  
    return response

ownertype_result=pd.DataFrame(columns=["author_id","gpt_otype"])
for index, row in ses_df[1101:].iterrows():
    response = ownertype_gpt4(row)
    answer = response['choices'][0]['message']['content']
    ownertype_result.loc[ownertype_result.shape[0],]=[row["author_id"],answer]
    time.sleep(1)
    print(index)
    
ownertype_result["gpt_otype"]=ownertype_result["gpt_otype"].str.lower()
ownertype_result[ownertype_result["gpt_otype"]=="personal"]
ownertype_result[ownertype_result["gpt_otype"]=="organizational"]

df_withotype=pd.merge(df_withsent,ownertype_result,on="author_id",how="left")
df_withotype.to_csv(r"output\full_withotype_230916_2.csv",index=False)

ses_df=pd.merge(ses_df,ownertype_result,on="author_id",how="left")
personal_df=ses_df[ses_df["gpt_otype"]=="personal"].reset_index(drop=True)
org_df=ses_df[ses_df["gpt_otype"]=="organizational"].reset_index(drop=True)

#guess job
def occupation_gpt4(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name \"{accid}\", and it is an \"{verified}\" account. Account description is \"{description}\". Does the owner introduce his job among those information? reply job name only. reply \"not sure\" if it is hard to determine. no explanation."
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        request_timeout=15
    )  
    return response

def soc2018_gpt(row):
    jobname = row["gpt_occupation"]
    prompt = f"What is the class that job \"{jobname}\ belongs to in the U.S Standard Occupational Classification System 2018? Use the highest level classification with 23 categories in this system. Answer only the full name of the classification. No explanation."
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        request_timeout=15
    )  
    return response

occupation_result=[]
for index, row in personal_df[617:].iterrows():
    response = occupation_gpt4(row)
    answer = response['choices'][0]['message']['content']
    occupation_result+=[answer]
    time.sleep(1)
    print(index)
    
personal_df["gpt_occupation"]=occupation_result
personal_df["gpt_occupation"]=personal_df["gpt_occupation"].str.lower()
personal_df.loc[personal_df["gpt_occupation"]=="not sure.","gpt_occupation"]="not sure"
personal_df["gpt_occupation"].unique()

soc_result=[]

for index, row in personal_df[781:].iterrows():
    if row["gpt_occupation"]=="not sure":
        soc_result+=["not sure"]
        print(index)
        continue
    response = soc2018_gpt(row)
    answer = response['choices'][0]['message']['content']
    soc_result+=[answer]
    time.sleep(1)
    print(index)

personal_df["gpt_occ_group"]=soc_result
personal_df.to_csv(r"output\full_personal_230916.csv")
personal_df["gpt_occ_group"].unique()

#orgnizational

def orgtype_gpt(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. What is the type of this orgnaization? Reply with one from Government,Institution, Enterprise, Nonprofit, Media, Others,or not sure. no explanation."
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        request_timeout=15
    )  
    return response


orgtype_result=[]
for index, row in org_df[409:].iterrows():
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    response = orgtype_gpt(row)
    answer = response['choices'][0]['message']['content']
    orgtype_result+=[answer]
    time.sleep(1)
    print(index)
    
org_df["gpt_org_type"]=orgtype_result
org_df.loc[org_df["gpt_org_type"]=="Insititution","gpt_org_type"]="Institution"
org_df.loc[org_df["gpt_org_type"]=='Nonprofit, Media',"gpt_org_type"]="Media"
org_df.to_csv(r"output\full_org_230916.csv")    
    

#keyword extraction
def keyword_gpt(tweet):
    prompt = (f"Extract keywords from this tweet: \"{tweet}\". Only reply the keywords, no explanation.")
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        request_timeout=15
    )  
    return response


keyword_result=[]
for index, row in df_withotype[1996:].iterrows():
    tweet = row["text"]
    response = keyword_gpt(tweet)
    answer = response['choices'][0]['message']['content']
    keyword_result+=[answer]
    print(index)

df_withotype['gpt_keywords']=keyword_result

keyword_gpt_df=sample_df[["Keyword","gpt_keywords"]]

