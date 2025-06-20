# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:34:53 2023

@author: TwoYoung
"""


import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score,precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

sample_df=pd.read_csv(r'')

#Sentiment Analysis Result

# Use NLTK VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Initialize VADER
sia = SentimentIntensityAnalyzer()

def vader_sentiment(tweet):
    tweet=str(tweet)
    sentiment_score = sia.polarity_scores(tweet)['compound']
    
    if sentiment_score >= 0.05:
        return "support"
    elif sentiment_score <= -0.05:
        return "oppose"
    else:
        return "neutral"
    

sample_df["nlp_sent"]=sample_df["text"].apply(vader_sentiment)

#draw confusion matrix
conf_matrix_tweet = confusion_matrix(sample_df["Opinion"],sample_df["gpt_sent_response"],labels=["support", "oppose", "neutral"])

conf_matrix_tweet_vader = confusion_matrix(sample_df["Opinion"],sample_df["nlp_sent"],labels=["support", "oppose", "neutral"])    


# Plotting the confusion matrix
vmin=min(conf_matrix_tweet_vader.min(),conf_matrix_tweet.min())
vmax=max(conf_matrix_tweet_vader.max(),conf_matrix_tweet.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.set(font_scale=1.2)
sns.heatmap(conf_matrix_tweet_vader, annot=True, fmt='g', cmap='Blues',vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[0])
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('Mannually Labelled Class')
axes[0].set_title('Confusion Matrix for NLTK-VADER Method')

sns.heatmap(conf_matrix_tweet, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[1])
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('Mannually Labelled Class')
axes[1].set_title('Confusion Matrix for GPT-4 Method')

#Owner Type
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

sample_df["nlp_otype"]=sample_df["name"].apply(get_ner_ent)
accuracy_score(sample_df["OwnerType"], sample_df["nlp_otype"])
precision_recall_fscore_support(sample_df["OwnerType"], sample_df["nlp_otype"], average='weighted')
accuracy_score(sample_df["OwnerType"], sample_df["gpt_otype"])
precision_recall_fscore_support(sample_df["OwnerType"], sample_df["gpt_otype"], average='weighted')


#Gender
import gender_guesser.detector as gender
gender_d= gender.Detector()
def nlp_gender_guesser(name):
    gender= gender_d.get_gender(name)
    return gender

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
    
sample_df.loc[sample_df['OwnerType']=='personal',"spacy_name"]=sample_df.loc[sample_df['OwnerType']=='personal','name'].apply(get_person_name)

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

name_sep=sample_df.loc[sample_df['OwnerType']=='personal','spacy_name'].apply(lambda x:pd.Series(separate_first_last_name(x)))  
sample_df.loc[sample_df['OwnerType']=='personal','first_name']= name_sep.loc[:,0]
sample_df.loc[sample_df['OwnerType']=='personal','last_name']= name_sep.loc[:,1]

sample_df.loc[sample_df['OwnerType']=='personal',"nlp_gender"]=sample_df.loc[sample_df['OwnerType']=='personal','first_name'].apply(nlp_gender_guesser)

sample_df.loc[sample_df['nlp_gender']=='unknown',"nlp_gender"]='not sure'
sample_df.loc[sample_df['nlp_gender']=='mostly_female',"nlp_gender"]='female'
sample_df.loc[sample_df['nlp_gender']=='mostly_male',"nlp_gender"]='male'
sample_df.loc[sample_df['nlp_gender']=='andy',"nlp_gender"]='not sure'

accuracy_score(sample_df.loc[sample_df['OwnerType']=='personal',"gender"], sample_df.loc[sample_df['OwnerType']=='personal',"nlp_gender"])
precision_recall_fscore_support(sample_df.loc[sample_df['OwnerType']=='personal',"gender"], sample_df.loc[sample_df['OwnerType']=='personal',"nlp_gender"], average='weighted')

sample_df.loc[(sample_df['OwnerType']=='personal') & (sample_df['gpt_gender'].isna()),"gpt_gender"]='not sure'
accuracy_score(sample_df.loc[sample_df['OwnerType']=='personal',"gender"], sample_df.loc[sample_df['OwnerType']=='personal',"gpt_gender"])
precision_recall_fscore_support(sample_df.loc[sample_df['OwnerType']=='personal',"gender"], sample_df.loc[sample_df['OwnerType']=='personal',"gpt_gender"], average='weighted')


#ethnicity
import ethnicolr as etr
#create a copy because ehnicolr package would remove empty records
temp_df=sample_df[sample_df['OwnerType']=='personal'].copy()
temp_df = etr.pred_census_ln(temp_df,'last_name',year=2010)

sample_df=pd.merge(sample_df, temp_df[["id_x","race"]],left_on='id_x',right_on='id_x',how='left')
del temp_df

sample_df=sample_df.rename(columns={"race":"nlp_ethnicity"})
#rewrite api to asian and hispanic to hispanic and latino, also remove speculation on empty or one word last name
sample_df.loc[sample_df['nlp_ethnicity']=='api','nlp_ethnicity']='asian'
sample_df.loc[sample_df['nlp_ethnicity']=='hispanic','nlp_ethnicity']='hispanic or latino'
sample_df.loc[(sample_df['OwnerType']=='personal') & (sample_df['nlp_ethnicity'].isna()),'nlp_ethnicity']='not sure'
sample_df.loc[(sample_df['OwnerType']=='personal') & (sample_df['last_name'].str.len()<=1),'nlp_ethnicity']='not sure'

accuracy_score(sample_df.loc[sample_df['OwnerType']=='personal',"ethnicity"], sample_df.loc[sample_df['OwnerType']=='personal',"nlp_ethnicity"])
precision_recall_fscore_support(sample_df.loc[sample_df['OwnerType']=='personal',"ethnicity"], sample_df.loc[sample_df['OwnerType']=='personal',"nlp_ethnicity"], average='weighted')

sample_df.loc[(sample_df['OwnerType']=='personal') & (sample_df['gpt_ethnicity'].isna()),"gpt_ethnicity"]='not sure'
accuracy_score(sample_df.loc[sample_df['OwnerType']=='personal',"ethnicity"], sample_df.loc[sample_df['OwnerType']=='personal',"gpt_ethnicity"])
precision_recall_fscore_support(sample_df.loc[sample_df['OwnerType']=='personal',"ethnicity"], sample_df.loc[sample_df['OwnerType']=='personal',"gpt_ethnicity"], average='weighted')

#Occupation Group
sample_df.loc[(sample_df['OwnerType']=='personal') & (sample_df['gpt_occ_group'].isna()),"gpt_occ_group"]='not sure'
accuracy_score(sample_df.loc[sample_df['OwnerType']=='personal',"Occupation"], sample_df.loc[sample_df['OwnerType']=='personal',"gpt_occ_group"])
precision_recall_fscore_support(sample_df.loc[sample_df['OwnerType']=='personal',"Occupation"], sample_df.loc[sample_df['OwnerType']=='personal',"gpt_occ_group"], average='weighted')


personal_df=sample_df[sample_df['OwnerType']=='personal']
matched=personal_df[personal_df["Occupation"]==personal_df["gpt_occ_group"]]
unmatched=personal_df[personal_df["Occupation"]!=personal_df["gpt_occ_group"]]
review_df=unmatched[['name','description','Occupation','gpt_occ_group']]

#Organization Type
sample_df.loc[(sample_df['OwnerType']=='organizational') & (sample_df['gpt_org_type'].isna()),"gpt_org_type"]='not sure'
sample_df.loc[(sample_df['OwnerType']=='organizational') & (sample_df['OrganizationalType']=='Not Sure'),'OrganizationalType']='not sure'

accuracy_score(sample_df.loc[sample_df['OwnerType']=='organizational',"OrganizationalType"], sample_df.loc[sample_df['OwnerType']=='organizational',"gpt_org_type"])
precision_recall_fscore_support(sample_df.loc[sample_df['OwnerType']=='organizational',"OrganizationalType"], sample_df.loc[sample_df['OwnerType']=='organizational',"gpt_org_type"], average='weighted')


org_df=sample_df[sample_df['OwnerType']=='organizational']
matched=org_df[org_df["OrganizationalType"]==org_df["gpt_org_type"]]
unmatched=org_df[org_df["OrganizationalType"]!=org_df["gpt_org_type"]]
review_df=unmatched[['name','description','OrganizationalType','gpt_org_type']]

#Keyword Extraction
def removeurlandat(tweet):
    tweet=str(tweet)
    wordlist=tweet.split(",")
    for xword in ['#','@','https','uhttps']:
        wordlist=list(word.strip() for word in wordlist if word.find(xword)==-1)  
    newtweet=",".join(wordlist)
    return newtweet



#RAKE Keywords
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from rake_nltk import Rake
import random
import spacy

r = Rake()
nlp = spacy.load("en_core_web_sm")

def take_five_word(word):
    word_list=word.split(",")
    if len(word_list)<=5:
        return word
    else:
        return ",".join(word_list[0:5])
    
def rake_keyword(text):
    text = str(text)
    text = re.sub(r'@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    

    # Extract keywords
    r.extract_keywords_from_text(text)
    wordlist = [x.split(" ") for x in r.get_ranked_phrases()]

    # Combine list and remove any residual symbols
    combined_list = [item for sublist in wordlist for item in sublist if item.isalpha()]
    combined_list=list(dict.fromkeys(combined_list))[:5]
    return ",".join(combined_list)

def spacy_keyword(text):
    # Process the text with spaCy
    text = str(text)
    text = re.sub(r'@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)

    # Extract entities and nouns/adjectives as keywords
    keywords = set()
    for entity in doc.ents:
        keywords.add(entity.text)

    for token in doc:
        if token.pos_ in ["NOUN", "ADJ"]:
            keywords.add(token.text)

    return ",".join(keywords)


# def generate_keywords(row):
#     if not isinstance(row['text'], str):
#         print(row['text'])
#         return ""
    
#     if (row['text'].isnumeric())|(len(row['text'])<=2):
#         print(row['text'])
#         return ""
#     gpt_keywords=[word.strip().lower() for word in row['gpt_keywords'].split(",") if word!=""]
#     nlp_keywords=[word.strip().lower() for word in row['nlp_keywords'].split(",") if word!=""]
    
#     keywords_both=list(set(gpt_keywords) & set(nlp_keywords))
#     keywords_gpt_only=list(set(gpt_keywords)-set(nlp_keywords))
#     keywords_no_gpt=list(set(nlp_keywords)-set(gpt_keywords))
    
#     if len(keywords_both)>=5:
#         return ",".join(keywords_both[0:5])
      
#     # fill keywords to 5
#     need_word_n=5-len(keywords_both)
    
#     if need_word_n>len(keywords_gpt_only):
#         keywords_response=keywords_both+keywords_gpt_only
#         need_word_n=5-len(keywords_response)
#         if need_word_n>len(keywords_no_gpt):
#             keywords_response=keywords_response+keywords_no_gpt
#         else:
#             more_word = random.sample(keywords_no_gpt, need_word_n)
#             keywords_response=keywords_response+more_word  
#     else:
#         if len(keywords_no_gpt)==0:
#             more_word = keywords_gpt_only[0:need_word_n]
#             keywords_response=keywords_both+more_word
#         else:
#             more_word = keywords_gpt_only[0:need_word_n-1] + keywords_no_gpt[0:1]
#             keywords_response=keywords_both+more_word
    
#     return ",".join(keywords_response)

def generate_keywords(row):
    if not isinstance(row['text'], str):
        print(row['text'])
        return ""
    
    if (row['text'].isnumeric())|(len(row['text'])<=2):
        print(row['text'])
        return ""
    gpt_keywords=[word.strip().lower() for word in row['gpt_keywords'].split(",") if word!=""]
    nlp_keywords=[word.strip().lower() for word in row['nlp_keywords'].split(",") if word!=""]
    
    keywords_both=list(set(gpt_keywords) & set(nlp_keywords))
    keywords_outsider=list((set(gpt_keywords)|set(nlp_keywords))-set(keywords_both))
    
    if len(keywords_both)>=5:
        return ",".join(keywords_both[0:5])
      
    # fill keywords to 5
    need_word_n=5-len(keywords_both)
    
    if need_word_n>len(keywords_outsider):
        keywords_response=keywords_both+keywords_outsider     
    else:
        more_word = random.sample(keywords_outsider,need_word_n)
        keywords_response=keywords_both+more_word
    
    return ",".join(keywords_response)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

#change keyword to word lists
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




sample_df=pd.read_csv(r'output\sample_all_231129.csv')
sample_df['gpt_keywords']=sample_df['gpt_keywords'].apply(removeurlandat)

sample_df['gpt_keywords']=sample_df['gpt_keywords'].apply(take_five_word)

sample_df['nlp_keywords']=sample_df["text"].apply(rake_keyword)
sample_df['spacy_keywords']=sample_df["text"].apply(spacy_keyword)
sample_df.loc[~sample_df['Keyword'].isna(),'n_keywords']=sample_df.loc[~sample_df['Keyword'].isna(),'Keyword']
sample_df.loc[sample_df['Keyword'].isna(),'n_keywords']=sample_df[sample_df['Keyword'].isna()].apply(generate_keywords,axis=1)


sample_df["Keyword"]=sample_df["Keyword"].apply(clean_words)
sample_df["gpt_keywords"]=sample_df["gpt_keywords"].apply(clean_words)
sample_df["nlp_keywords"]=sample_df["nlp_keywords"].apply(clean_words)


sample_df["match"]=sample_df.apply(compare_word_lists,args=("Keyword","nlp_keywords"),axis=1)
sample_df["match"].sum()

sample_df[sample_df['nlp_keywords'].duplicated()].index.to_list()

tt=sample_df[sample_df['nlp_keywords'].duplicated()].index.to_list()
tt=[n+1 for n in tt]


sample_df.to_csv(r'output\sample_all_231201.csv',index=False)
