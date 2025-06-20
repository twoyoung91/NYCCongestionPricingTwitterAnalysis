# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:44:13 2023

@author: TwoYoung
"""

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

word="Yang Yang. Unverified. PhD Candidate, research assistant @ UNC, transportation planner, reporter"

text=NER(word)
person_count=0
org_count=0
for ent in text.ents:
    print(ent.label_)
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
