# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:42:25 2024

@author: TwoYoung
"""

import os
import re
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import seaborn as sns
import time



import google.generativeai as genai
import os

token = 'enter_your_token_here'  # Replace with your actual token

export API_KEY='enter_your_token_here'  # Replace with your actual token
    
genai.configure(api_key=token)

model = genai.GenerativeModel('gemini-1.5-flash')

response = model.generate_content("Write a story about an AI and magic")

print(response.text)


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

def get_answer(response):
    answer = response['choices'][0]['message']['content']
    return answer

#speculate personal/organizational
def ownertype_gemini(row):
    accid = row["name"]
    verified = row["veriacc"]
    description = row["description"]
    prompt = f"This is the Twitter account name '\"{accid}\"', and it is an \"{verified}\" account. Account description is '\"{description}\"'. Is this a personal account or an organizational account? Answer this question with personal, organizational, or not sure without explanation"
    messages =[{"role":"assistant","content":prompt}]
    response = model.generate_content(prompt)
    return response

# #test case
# user = sample_df.loc[0,]
# response = ownertype_gemini(user)
# answer=response.text
# print(answer)

ownertype_result=[]
for index, row in sample_df.iterrows():
    response = ownertype_gemini(row)
    answer = response.text
    ownertype_result+=[answer]
    time.sleep(2)
    print(index)
