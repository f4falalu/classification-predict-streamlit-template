#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 02:03:08 2022

@author: fbarde
"""

import numpy as np
import pickle


#loaded model
loaded_model = pickle.load(open("/Users/fbarde/Desktop/classify/trained_model.sav", "rb"))





tweet_input = [input('enter a text: ')]
prediction=loaded_model.predict(tweet_input)

if prediction < 0:
    print("Negative")
elif prediction == 0:
    print("Neutral")
else:
    print("Positive")
