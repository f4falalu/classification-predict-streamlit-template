#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:24:51 2022

@author: fbarde
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
import streamlit_wordcloud as wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import re
import time
import os


#loaded model
loaded_model = pickle.load(open("/Users/fbarde/Desktop/classify/TM12_1.0_LogReg_model.pkl", "rb"))

#load dataset


#creating a function for prediction

def tweet_prediction(tweet_input):
    
    #tweet_input = []
    
    #tweet_input = [input('enter a text: ')]
    prediction=loaded_model.predict(tweet_input)

    if prediction < 0:
        return "Negative"
    elif prediction == 0:
        return "Neutral"
    else:
        return "Positive"

#creating WordCloud
stopwords = set(STOPWORDS)

extra_stopwords = ["The", "It", "it", "in", "In", "wh","rt"]

def processed_text(message):
  message = re.sub("https?:\/\/\S+", "", message)  # replacing url with domain name
  message = re.sub("#[A-Za-z0–9]+", " ", message)  # removing #mentions
  message = re.sub("#", " ", message)  # removing hash tag
  message = re.sub("\n", " ", message)  # removing \n
  message = re.sub("@[A-Za-z0–9]+", "", message)  # removing @mentions
  message = re.sub("RT", "", message)  # removing RT
  message = re.sub("^[a-zA-Z]{1,2}$", "", message)  # removing 1-2 char long words
  message = re.sub("\w*\d\w*", "", message)  

  for word in extra_stopwords:
        message = message.replace(word, "")
        
        message = message.lower()
    # will split and join the words
        message=' '.join(message.split())
  return message

#creating sentiment ratio
def getPolarity(message):
    sentiment_polarity = TextBlob(message).sentiment.polarity
    return sentiment_polarity
#convert polarity to sentiment
def getAnalysis(polarity_score):
    if polarity_score < 0:
        return "Negative"
    elif polarity_score == 0:
        return "Neutral"
    else:
        return "Positive"
    
#convert sentiment 
######################################################
    
def main():
    
    #title for streamlit
    st.title('Tweet Classification Prediction')
    
    ######### body container ############

    
    ######### Side Bar #################
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")
        
    
    # Building out the predication page
    if selection == "Prediction":
         
         
         
         #input data field
         message = st.text_input("Type a Message")
         
         #code for prediction
         tweet = ''
         #button for classification
         if st.button("Classify"):
             st.info("Prediction with ML Models")
             tweet = tweet_prediction([message])
             st.success(tweet)
             
             #word cloud
             st.info("Word Cloud")
             
             wordcld = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(message)
             st.set_option('deprecation.showPyplotGlobalUse', False)
             plt.imshow(wordcld)
             plt.xticks([])
             plt.yticks([])
             st.pyplot()
             
             #Sentiment Ratio
             st.info("Sentiment Ratio")
             message_sentiment = pd.DataFrame(["message"]).apply(processed_text)
             #message_sentiment = message_sentiment["message"].apply(processed_text)
             
             
             
             
             
             
         #st.success(selection)
         
    
   
    
    
    
    
    
    
    
    
        
        
    
 
    
    
    
    
   
  
  
   
    

if __name__ == '__main__':
    main()

    