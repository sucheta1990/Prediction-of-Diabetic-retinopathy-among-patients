#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from pickle import load
import pickle
import numpy as np


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


st.title('Ratinopathy Prediction in Patients')
st.sidebar.header('User Input Parameters')


# In[4]:


st.subheader("MODEL DEPLOYMENT:Logistic Regression")


# In[ ]:


st.image('club.JPG')


# In[ ]:


loaded_model=pickle.load(open('LOG_deploy.save','rb'))


# In[ ]:


def user_input_features():
    Age =st.sidebar.number_input ('Enter age')
    Systolic_BP=st.sidebar.number_input("enter the Systolic BP")
    Diastolic_BP=st.sidebar.number_input('enter the Diastolic BP')
    Cholesterol=st.sidebar.number_input('enter the Cholesterol')
    Ratinopathy_prediction={'Age':Age,
           'Systolic BP':Systolic_BP,
           'Diastolic BP':Diastolic_BP,
           'Cholesterol':Cholesterol}
    features = pd.DataFrame(Ratinopathy_prediction,index = [0])
    return features


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[ ]:


import pickle
loaded_model=pickle.load(open('LOG_deploy.save','rb'))
loaded_model


# In[ ]:


prediction = loaded_model.predict(df)

prediction_proba = loaded_model.predict_proba(df)


# In[ ]:


st.subheader('Prediction Probability')
prediction_proba=pd.DataFrame(prediction_proba)
b=[0]
a= prediction_proba.iloc[:,0]
if (a>0.7).any():
    a='Not Suffering from Ratinopathy'
    st.write(a)
else:
    b='Sufferring from Ratinopathy'
    st.write(b)

st.write(prediction_proba)


# In[ ]:


output=pd.concat([df,prediction_proba],axis=1)

