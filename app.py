# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:20:31 2020
@author: Krish Naik
"""
# -*- coding: utf-8 -*-
"""

@author: ADITYA AGARWAL
"""

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import pickle
#from flasgger import Swagger
from tensorflow.keras.models import load_model
import streamlit as st 
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

#app=Flask(__name__)
#Swagger(app)

model = load_model('model.h5')
index_to_class = open("index_to_class.pkl","rb")
tokenizer = open("tokenizer.pkl","rb")
index_to_class=pickle.load(index_to_class)
tokenizer=pickle.load(tokenizer)

#@app.route('/')
def welcome():
    return "Welcome All"

def padding_seq(tokenizer,seq):
  seq=tokenizer.texts_to_sequences(seq)
  padded_seq=pad_sequences(seq,truncating='post',padding='post',maxlen=50)
  return padded_seq

#@app.route('/predict',methods=["Get"])
def predict_mood(s):
  s=s.lower()  
  s=[s]
  test=padding_seq(tokenizer,s)
  y=np.argmax(test,axis=-1)
  return index_to_class[y[0]]


def main():
    st.title("HAPPY BOT")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit HAPPY BOT ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    s = st.text_input("YOUR THOUGHTS","Type Here")
    #skewness = st.text_input("skewness","Type Here")
    #curtosis = st.text_input("curtosis","Type Here")
    #entropy = st.text_input("entropy","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_mood(s)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    