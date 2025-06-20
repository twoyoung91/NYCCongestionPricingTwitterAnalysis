# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:39:00 2023

@author: TwoYoung
"""
from openai import OpenAI
import os
import re
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import seaborn as sns
import time


# Set up the OpenAI API client
openai.api_key = "Your API Key Here"

client = OpenAI(api_key="Your API Key Here")

sample_df=pd.read_csv("sample_tweets_100_3.csv")

#remove same content tweets
def removeurl(tweet):
    wordlist=tweet.split()
    for xword in ['https','uhttps']:
        wordlist=list(word for word in wordlist if word.find(xword)==-1)  
    newtweet=" ".join(wordlist)
    return newtweet

sample_df['text']=sample_df['text'].apply(removeurl)
sample_df.loc[sample_df["verified"]==True,"veriacc"]="verified"
sample_df.loc[sample_df["verified"]==False,"veriacc"]="unverified"

def get_answer(response):
    answer = response['choices'][0]['message']['content']
    return answer


#speculate personal/organizational
def ownertype_gpt(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. Is this a personal account or an organizational account? Answer this question with personal, organizational, or not sure without explanation"
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
        timeout=15,
    )  
    return response

user = sample_df2.loc[0,]
response = ownertype_gpt4(user)
answer = response['choices'][0]['message']['content']
print(answer)

ownertype_result=[]
for index, row in sample_df2.iterrows():
    response = ownertype_gpt4(row)
    answer = response['choices'][0]['message']['content']
    ownertype_result+=[answer]
    time.sleep(2)
    print(index)
    
sample_df["gpt_otype"]=ownertype_result
sample_df["gpt_otype"]=sample_df["gpt_otype"].str.lower()
sample_df["OwnerType"]=sample_df["OwnerType"].str.lower()
sample_df.loc[sample_df2["gpt_otype"]=="not sure.","gpt_otype"]="not sure"
sample_df["gpt_otype"].unique()

matched=sample_df[sample_df["gpt_otype"]==sample_df["OwnerType"]]
sample_df.to_csv(r"output\ownertype_230913_2.csv")
sample_df=pd.read_csv(r"output\ownertype_230913_3.csv")

#spaCy name entity check
import spacy
from spacy import displacy
NER=spacy.load('en_core_web_sm')

def get_ner_ent(word):
    text=NER(word)
    person_count=0
    org_count=0
    for ent in text.ents:
        if ent.label_=="PERSON":
            person_count+=1
        elif ent.label_!="":
            org_count+=1      
    
    if person_count==0 and org_count==0:    
        return "not sure"
    elif person_count>org_count:
        return "personal"
    else:
        return "organizational"

sample_df["spacy_otype"]=sample_df2["name"].apply(get_ner_ent)
matched=sample_df[sample_df["spacy_otype"]==sample_df["OwnerType"]]
unmatched=sample_df[sample_df["space_otype"]!=sample_df["OwnerType"]]
sample_df.to_csv(r"output\ownertype_230913_3.csv")


#speculate jobs
sample_df=pd.read_csv(r"output\ses_sample_2.csv")
sample_df.loc[sample_df["verified"]==True,"veriacc"]="verified"
sample_df.loc[sample_df["verified"]==False,"veriacc"]="unverified"

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
        timeout=15,
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
        timeout=15,
    )  
    return response

personal_df=sample_df[sample_df["OwnerType"]=="Personal"].reset_index()
row = personal_df.loc[6,]
response = soc2018_gpt(row)
answer = response['choices'][0]['message']['content']
print(answer)

occupation_result=[]

for index, row in personal_df.iterrows():
    response = occupation_gpt4(row)
    answer = response['choices'][0]['message']['content']
    occupation_result+=[answer]
    time.sleep(2)
    print(index)
    
personal_df["gpt_occupation"]=occupation_result
personal_df["gpt_occupation"]=personal_df["gpt_occupation"].str.lower()
personal_df["Occupation"]=personal_df["Occupation"].str.lower()
personal_df.loc[personal_df["gpt_occupation"]=="not sure.","gpt_occupation"]="not sure"
personal_df["gpt_occupation"].unique()

soc_result=[]

for index, row in personal_df.iterrows():
    if row["gpt_occupation"]=="not sure":
        soc_result+=["not sure"]
        print(index)
        continue
    response = soc2018_gpt(row)
    answer = response['choices'][0]['message']['content']
    soc_result+=[answer]
    time.sleep(2)
    print(index)

personal_df["gpt_occ_group"]=soc_result
personal_df.to_csv(r"output\occupation_230915_5.csv")



matched=personal_df[personal_df["gpt_occ_group"]==personal_df["OccType"]]
unmatched=personal_df[personal_df["gpt_occ_group"]!=personal_df["OccType"]]
tt=unmatched[["name","description","Occupation","OccType","gpt_occupation","gpt_occ_group"]]


#speculate gender of person
all_df=pd.read_csv(r'output\full_all_230916_2.csv')
all_df=all_df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
personal_df=all_df[all_df["gpt_otype"]=="personal"].reset_index()

#NLP Method
#take names out for each personal account
def get_person_name(word):
    doc = NER(word)
    names=[]
    if len(doc.ents)>0:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names+=[ent.text]
        if names:
            return names[0]
    
    #when spacy can't capture names, use the orignal string after removing hashtags
    doc =[token for token in  re.split(r'[ .|,_]',word) if token]
    if doc:
        doc = [token for token in doc if token[0]!="#" and token[0]!="@"]
        return " ".join(doc)
    else:
        return ""

personal_df['spacy_name']=personal_df['name'].apply(get_person_name)

def separate_first_last_name(name):
    if len(name)==0:
        return "",""
    doc = [token for token in  re.split(r'[ .|,_]',name) if token]
    if len(doc)==0:
        return "",""
    elif len(doc)==1:
        return doc[0],""
    else:
        return doc[0],doc[-1]
    
personal_df[['first_name','last_name']]=personal_df['spacy_name'].apply(lambda x:pd.Series(separate_first_last_name(x)))


# speculate gender and ethnicity
import gender_guesser.detector as gender
import ethnicolr as etr

gender_d= gender.Detector()

def nlp_gender_guesser(name):
    gender= gender_d.get_gender(name)
    return gender

personal_df["nlp_gender"]=personal_df["first_name"].apply(nlp_gender_guesser)

#create a copy because ehnicolr package would remove empty records
temp_df=personal_df.copy()
temp_df = etr.pred_census_ln(temp_df,'last_name',year=2010)

personal_df=pd.merge(personal_df, temp_df[["id_x","race"]],left_on='id_x',right_on='id_x',how='left')
del temp_df

personal_df=personal_df.drop("index",axis=1)
personal_df=personal_df.rename(columns={"race":"nlp_ethnicity"})
#rewrite api to asian and hispanic to hispanic and latino, also remove speculation on empty or one word last name
personal_df.loc[personal_df['nlp_ethnicity']=='api','nlp_ethnicity']='asian'
personal_df.loc[personal_df['nlp_ethnicity']=='hispanic','nlp_ethnicity']='hispanic or latino'
personal_df.loc[personal_df['nlp_ethnicity'].isna(),'nlp_ethnicity']='not sure'
personal_df.loc[personal_df['last_name'].str.len()<=1,'nlp_ethnicity']='not sure'
personal_df.loc[personal_df['nlp_gender']=="unknown",'nlp_gender']='not sure'


#check accuracy
#gender
matched=personal_df[personal_df["gender"]==personal_df["nlp_gender"]]
unmatched=personal_df[personal_df["gender"]!=personal_df["nlp_gender"]]

matched=personal_df[personal_df["ethnicity"]==personal_df["nlp_ethnicity"]]
unmatched=personal_df[personal_df["ethnicity"]!=personal_df["nlp_ethnicity"]]

#GPT Method
def gender_gpt(row):
    accid = row["name"]
    username = row["username"]
    description = row["description"]
    prompt = f"This is the Twitter account name '{accid}', and its username is '{username}'. try your best to guess the most likely gender of the owner of this account. Answer gender with male or female. Answer not sure if you don't have any information to speculate. No explanation."
    messages =[{"role":"assistant","content":prompt}]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        timeout=15,
    )  
    return response

gender_result=[]

for index, row in personal_df.iterrows():
    response = gender_gpt(row)
    answer = response.choices[0].message.content
    gender_result+=[answer]
    time.sleep(1)
    print(index)
    
personal_df["gpt_gender"]=gender_result
personal_df["gpt_gender"]=personal_df["gpt_gender"].str.lower()
personal_df.loc[personal_df["gpt_gender"]=="Not sure.","gpt_gender"]="not sure"
personal_df["gpt_gender"].unique()


# def ethnicity_gpt(row):
#     accid = row["name"]
#     username = row["username"]
#     description = row["description"]
#     prompt = f"This is the Twitter account name '{accid}', and its username is '{username}'. try your best to guess the most likely ethnicity of the owner of this account.  Answer ethnicity with 'White', 'Hispanic or Latino', 'Black', 'Asian, Native Hawaiian and Other Pacific Islander', 'American Indian and Alaska Native'. Answer not sure if you don't have any information to speculate. No explanation."
#     messages =[{"role":"assistant","content":prompt}]
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=messages,
#         temperature=0.2,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         timeout=15,
#     )  
#     return response


def ethnicity_gpt(row):
    accid = row["name"]
    username = row["username"]
    description = row["description"]
    prompt = f"The account is named '{accid}', with the username '{username}'. What is its most possible ethnic origin? Answer with 'White', 'Hispanic or Latino', 'Black', 'Asian, Native Hawaiian and Other Pacific Islander', 'American Indian and Alaska Native'. Answer not sure if you don't have a name to process. No explanation."
    messages =[{"role":"assistant","content":prompt}]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        timeout=15,
    )  
    return response
ethnicity_result=[]

for index, row in personal_df.iterrows():
    response = ethnicity_gpt(row)
    answer = response.choices[0].message.content
    ethnicity_result+=[answer]
    time.sleep(1)
    print(index)
    
personal_df["gpt_ethnicity"]=ethnicity_result
personal_df["gpt_ethnicity"]=personal_df["gpt_ethnicity"].str.lower()
personal_df["gpt_ethnicity"] = personal_df["gpt_ethnicity"].str.replace("'", '')
personal_df.loc[personal_df["gpt_ethnicity"]=="asian, native hawaiian and other pacific islander","gpt_ethnicity"]="asian"
personal_df["gpt_ethnicity"].unique()


#check accuracy
#gender
matched=personal_df[personal_df["gender"]==personal_df["gpt_gender"]]
unmatched=personal_df[personal_df["gender"]!=personal_df["gpt_gender"]]

#ethnicity
matched=personal_df[personal_df["ethnicity"]==personal_df["gpt_ethnicity"]]
unmatched=personal_df[personal_df["ethnicity"]!=personal_df["gpt_ethnicity"]]

#personal_df.to_csv('output\ses_sample_genderethnicity.csv',index=False)

#Orgnaizaiton Speculation
org_df=sample_df[sample_df["OwnerType"]=="organizational"].reset_index()




def orgtype_gpt(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. What is the type of this orgnaization? Reply with one from Government,Insititution, Enterprise, Nonprofit, Media, Others,or not sure. no explanation."
    messages =[{"role":"assistant","content":prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        n=1,
        stop=None,
        timeout=15,
    )  
    return response

#test
row = org_df.loc[6,]
response = orgtype_gpt(row)
answer = response['choices'][0]['message']['content']


orgtype_result=[]
for index, row in org_df.iterrows():
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    response = orgtype_gpt(row)
    answer = response['choices'][0]['message']['content']
    orgtype_result+=[answer]
    time.sleep(2)
    print(index)
    

org_df["gpt_org_type"]=orgtype_result
org_df["gpt_org_type"]=org_df["gpt_org_type"].str.replace('"', '')
matched=org_df[org_df["gpt_org_type"]==org_df["OrganizationalType"]]
unmatched=org_df[org_df["gpt_org_type"]!=org_df["OrganizationalType"]]
tt=unmatched[["name","description","OrganizationalType","gpt_org_type"]]
