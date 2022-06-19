#!/usr/bin/env python
# coding: utf-8

# Pipeline, Naive_Bayes and CountVectorizer - Restaurant reviews by Customers

# Dataset from Kaggle:  https://www.kaggle.com/datasets/arthurdfr/trip-advisor-reviews

# Goal: Predict Future customer reviews to be positive or negative regarding the restaurant

# In[87]:


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
from sklearn.pipeline import Pipeline


# In[91]:


review_ta = pd.read_csv('Documents/reviews.csv')
review_ta.head()


# In[92]:


# eda
review_ta.describe()


# In[97]:


review_ta.isna().sum()
# no NaN values


# In[121]:


# split datra then change message to numerical
X_train, X_test, y_train, y_test = train_test_split(review_ta.review, review_ta.rate, test_size=0.3 )


# In[153]:


# set up pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('mnb', MultinomialNB(alpha=0.01))
])


# In[154]:


# train reviews using the vectorizer in the pipeline
clf.fit(X_train, y_train)


# In[155]:


# check the accuracy score of model
clf.score(X_test, y_test)


# In[163]:


cust_reviews = [
    "Ate there Friday nite, very poor service",
    "What a joke, hated the meal",
    "The food was amazing!!",
    "Beautiful decor, loved the appetizer",
    "Awesome steak, cooked to perfection!!",
    "disappointed"
]


# In[164]:


clf.predict(cust_reviews)


# In[ ]:


# prediction accuracy at 73%, all predicted correctly except 6th review. 
# Possibly, 1 word reviews not are good for prediction


# In[ ]:




