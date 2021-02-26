# -*- coding: utf-8 -*-
"""
Code to calculate tangentiality and on-topic measures from a speech excerpt, as in Morgan et al 2021:
https://doi.org/10.1101/2021.01.04.20248717
Please cite the paper above if you use this code for your own work.
Author: Dr Sarah E Morgan, 21/02/2021
"""


import numpy as np
from basic_meas import *
from coh_meas import *
from scipy import spatial
from sklearn.linear_model import LinearRegression
import sys

###### Use downloaded word2vec model: # https://github.com/eyaler/word2vec-slim
model = gensim.models.KeyedVectors.load_word2vec_format('c:/Users/sem91/Documents/Research/speech/Corpora/w2v_googlenews/GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
embedding_size = 300

# Concatenate the picture description with the participant's response
def comb_speech_pic(pictext,speechtext):
    text = pictext + ' . ' + speechtext # fullstop here should ensure that picture description is always treated as a separate sentence
    return text

def tangent_features(embedding):
    
    # Get picture description embedding vector:
    embed_pic = embedding[0]
    
    # Calculate cosine similarity:
    similarity_vec=[]
    for embed in embedding:
        similarity_vec.append(1 - spatial.distance.cosine(embed_pic, embed))
            
    similarity_vec = [x for x in similarity_vec if ~np.isnan(x)] # removes any nan entries (which occur if a sentence doesn't contain any word in the word embedding dictionary)
        
    # Fit linear regression and calculate slope:
    x = range(1,len(similarity_vec))
    x = np.array(x).reshape((-1,1)) # puts in right format for regression
    y = np.array(similarity_vec)
    y = np.delete(y,0)
    modelLR = LinearRegression().fit(x, y)
        
    dummy=modelLR.coef_
    tangent=dummy[0]
    ontopic=(sum(similarity_vec)-1)/(len(similarity_vec)-1)
    
    return tangent, ontopic

# Function to pull everything together, taking an input text and outputting tangentiality and on-topic score:
def meas_tangent(pictext,speechtext):
    
    nosent_pic=get_no_sent(pictext)
    nowords_speech=get_no_words(speechtext)
    
    if nosent_pic!=1:
        sys.exit("Error: picture description must contain exactly 1 sentence.")
    
    if nowords_speech==0:
        sys.exit("Error: no speech excerpt provided. Please provide an excerpt!")
    
    # start by concatenating picture description and participant's response:
    text=comb_speech_pic(pictext,speechtext)
    
    # Get processed, split sentences:
    sentences_split = text2procsent(text)
    
    
    if len(sentences_split)<2: # no speech sentences, only picture sentence (or neither)
        tangent_result = float('NaN'), float('NaN')
        print("Error: not contain enough non-stopwords in either speech and/or picture description for calculation.")
    else:
        embedding = sentence2vec(sentences_split, embedding_size, model)
        if np.nansum(np.nansum(embedding))==0: # if embedding is all zeros/NaNs, return NaNs.
            tangent_result = float('NaN'), float('NaN')
        elif len(sentences_split)==2: # only 1 speech sentence, plus picture sentence
            tangent_result = float('NaN'), tangent_features(embedding)[1]
            print("Error: only 1 speech sentence with enough non-stop words. Therefore tangentiality cannot be calculated.")
        else:
            tangent_result = tangent_features(embedding)   
    
    return tangent_result