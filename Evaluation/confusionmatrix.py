# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:58:09 2024

@author: TwoYoung
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score,precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

sample_df=pd.read_csv(r'output\sample_llama3_240724.csv')

#draw confusion matrix
conf_matrix_nlp = confusion_matrix(sample_df["Opinion"],sample_df["nlp_sent"],labels=["support", "oppose", "neutral"])    

conf_matrix_gpt = confusion_matrix(sample_df["Opinion"],sample_df["gpt_sent_response"],labels=["support", "oppose", "neutral"])

conf_matrix_claude = confusion_matrix(sample_df["Opinion"],sample_df["claude_sent"],labels=["support", "oppose", "neutral"])    

conf_matrix_qwen2 = confusion_matrix(sample_df["Opinion"],sample_df["qwen2_sent"],labels=["support", "oppose", "neutral"])    

conf_matrix_llama3 = confusion_matrix(sample_df["Opinion"],sample_df["llama3_sent"],labels=["support", "oppose", "neutral"])    



# Plotting the confusion matrix
vmin=min(conf_matrix_nlp.min(),conf_matrix_gpt.min(),conf_matrix_claude.min(),conf_matrix_qwen2.min(),conf_matrix_llama3.min())
vmax=max(conf_matrix_nlp.max(),conf_matrix_gpt.max(),conf_matrix_claude.max(),conf_matrix_qwen2.max(),conf_matrix_llama3.max())

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

sns.set(font_scale=1.2)
sns.heatmap(conf_matrix_nlp, annot=True, fmt='g', cmap='Blues',vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[0,0])
axes[0,0].set_xlabel('Predicted Class')
axes[0,0].set_ylabel('Mannually Labelled Class')
axes[0,0].set_title('Confusion Matrix for NLTK-VADER Model')

sns.heatmap(conf_matrix_gpt, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[0,1])
axes[0,1].set_xlabel('Predicted Class')
axes[0,1].set_ylabel('Mannually Labelled Class')
axes[0,1].set_title('Confusion Matrix for GPT-4 Model')

sns.heatmap(conf_matrix_claude, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[0,2])
axes[0,2].set_xlabel('Predicted Class')
axes[0,2].set_ylabel('Mannually Labelled Class')
axes[0,2].set_title('Confusion Matrix for Claude-3.5 Model')

sns.heatmap(conf_matrix_qwen2, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[1,0])
axes[1,0].set_xlabel('Predicted Class')
axes[1,0].set_ylabel('Mannually Labelled Class')
axes[1,0].set_title('Confusion Matrix for Qwen2-7b Model')

sns.heatmap(conf_matrix_llama3, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["support", "oppose", "neutral"], yticklabels=["support", "oppose", "neutral"], ax=axes[1,1])
axes[1,1].set_xlabel('Predicted Class')
axes[1,1].set_ylabel('Mannually Labelled Class')
axes[1,1].set_title('Confusion Matrix for Llama3.1-8b Model')

# Hide the unused subplot
axes[1, 2].set_visible(False)


# # Adjust spacing between the rows and columns
fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust vertical space and horizontal space

# Show the plot
plt.savefig("figures\confusionmatrix.svg")



#for sentiment 140 result
sen140_df=pd.read_csv(r'output/sentiment140/sent140_llama3_270724.csv')

#draw confusion matrix
conf_matrix_nlp = confusion_matrix(sen140_df["true_sent"],sen140_df["vader_sent"],labels=["positive", "negative", "neutral"])    

conf_matrix_gpt = confusion_matrix(sen140_df["true_sent"],sen140_df["gpt_sent_response"],labels=["positive", "negative", "neutral"])

conf_matrix_claude = confusion_matrix(sen140_df["true_sent"],sen140_df["claude_sent"],labels=["positive", "negative", "neutral"])    

conf_matrix_qwen2 = confusion_matrix(sen140_df["true_sent"],sen140_df["qwen2_sent"],labels=["positive", "negative", "neutral"])    

conf_matrix_llama3 = confusion_matrix(sen140_df["true_sent"],sen140_df["llama3_sent"],labels=["positive", "negative", "neutral"])    



# Plotting the confusion matrix
vmin=min(conf_matrix_nlp.min(),conf_matrix_gpt.min(),conf_matrix_claude.min(),conf_matrix_qwen2.min(),conf_matrix_llama3.min())
vmax=max(conf_matrix_nlp.max(),conf_matrix_gpt.max(),conf_matrix_claude.max(),conf_matrix_qwen2.max(),conf_matrix_llama3.max())

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

sns.set(font_scale=1.2)
sns.heatmap(conf_matrix_nlp, annot=True, fmt='g', cmap='Blues',vmin=vmin, vmax=vmax, xticklabels=["positive", "negative", "neutral"], yticklabels=["positive", "negative", "neutral"], ax=axes[0,0])
axes[0,0].set_xlabel('Predicted Class')
axes[0,0].set_ylabel('Mannually Labelled Class')
axes[0,0].set_title('Confusion Matrix for NLTK-VADER Model')

sns.heatmap(conf_matrix_gpt, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["positive", "negative", "neutral"], yticklabels=["positive", "negative", "neutral"], ax=axes[0,1])
axes[0,1].set_xlabel('Predicted Class')
axes[0,1].set_ylabel('Mannually Labelled Class')
axes[0,1].set_title('Confusion Matrix for GPT-4 Model')

sns.heatmap(conf_matrix_claude, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["positive", "negative", "neutral"], yticklabels=["positive", "negative", "neutral"], ax=axes[0,2])
axes[0,2].set_xlabel('Predicted Class')
axes[0,2].set_ylabel('Mannually Labelled Class')
axes[0,2].set_title('Confusion Matrix for Claude-3.5 Model')

sns.heatmap(conf_matrix_qwen2, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["positive", "negative", "neutral"], yticklabels=["positive", "negative", "neutral"], ax=axes[1,0])
axes[1,0].set_xlabel('Predicted Class')
axes[1,0].set_ylabel('Mannually Labelled Class')
axes[1,0].set_title('Confusion Matrix for Qwen2-7b Model')

sns.heatmap(conf_matrix_llama3, annot=True, fmt='g', cmap='Blues', vmin=vmin, vmax=vmax, xticklabels=["positive", "negative", "neutral"], yticklabels=["positive", "negative", "neutral"], ax=axes[1,1])
axes[1,1].set_xlabel('Predicted Class')
axes[1,1].set_ylabel('Mannually Labelled Class')
axes[1,1].set_title('Confusion Matrix for Llama3.1-8b Model')

# Hide the unused subplot
axes[1, 2].set_visible(False)


# # Adjust spacing between the rows and columns
fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust vertical space and horizontal space

# Show the plot
plt.savefig("figures\confusionmatrix_sent140.svg")

