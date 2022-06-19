#!/usr/bin/env python
# coding: utf-8

# Naive_Bayes and CountVectorizer - Restaurant reviews by Customers

# Dataset from Kaggle:  https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text

# Goal: Predict Future customer texts to be positive or negative regarding the restaurant

# In[38]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# In[43]:


sentiment = pd.read_csv('Documents/Emotion_final.csv')
sentiment.head()


# In[44]:


# Find Emotion column emotions
sentiment.groupby('Emotion').describe()


# In[82]:


def pos_emotion(x):
    if x == 'love':
        return 1
    elif x == 'happy':
        return 1
    else:
        return 0
    
sentiment['positive'] = sentiment['Emotion'].apply(pos_emotion)
sentiment.head()


# In[71]:


# split datra then change message to numerical
X_train, X_test, y_train, y_test = train_test_split(sentiment.Text, sentiment.positive, test_size=0.3 )


# In[72]:


# changing message data using CountVectorizer
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)
X_train_count.toarray()[0:3]


# In[74]:


model = MultinomialNB()
model.fit(X_train_count, y_train)


# In[75]:


# now trained and ready to make predictions


# In[81]:


customer_text = [
    "The hostess was rude",
    "I was delighted with dinner",
    "The dessert was divine!!",
    "The prices were so expensive, they are greedy"
]

customer_text_count = cv.transform(customer_text)
model.predict(customer_text_count)


# In[ ]:


# prediction results:  negative, positive, positive, negative customer reviews


# In[83]:


# measure accuracy score
X_test_count = cv.transform(X_test)
model.score(X_test_count, y_test)


# In[ ]:




