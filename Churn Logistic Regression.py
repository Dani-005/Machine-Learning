#!/usr/bin/env python
# coding: utf-8

# Logistic Regression Machine Learning - Restaurant Loyalty Customer Program to find Churn

# Dataset from Kaggle:  https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction    modified for use in restaurant example

# Goal:  Predict if a customer in loyalty program will churn

# In[129]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix


# In[130]:


churn = pd.read_csv('Documents/churn.csv')
churn.head()


# In[131]:


churn.shape


# In[132]:


churn.InLoyalty.unique()


# In[133]:


le = LabelEncoder()


# In[134]:


inloy = churn
inloy.InLoyalty = le.fit_transform(inloy.InLoyalty)
inloy.head()
# InLoyalty 0='No', 1='Yes'


# In[135]:


reserv = churn
reserv.Reservation = le.fit_transform(reserv.Reservation)
reserv.head()
# reservation 0='No', 1='Yes'


# In[136]:


churn.describe()


# In[137]:


ch = reserv
ch.head()


# Use age, InLoyalty Program and if they made a reservation to predict churn

# In[138]:


X = ch[['Age', 'InLoyalty', 'Reservation']]
y = ch['Churn']


# In[139]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[140]:


len(X_train)


# In[141]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)
# model has 83% accuracy to predict churn


# In[144]:


# predict a 55 yr old, InLoyalty program, made a reservation that they will remain a customer
logreg.predict([[55, 2, 1]])
# prediction no churn


# In[145]:


y_predicted = logreg.predict(X_test)


# In[147]:


# run a confusion matrix to see predictions
cm = confusion_matrix(y_test, y_predicted)
cm


# In[148]:


plt.figure(figsize=(5,3))
sb.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[149]:


print(ch.corr())
# review the correlation between features and target


# In[ ]:


# If person is in a loyalty program, predicted that they will stay in the program
# There is a negative correlation with churn and a reservation. 
# There is a small positive correlation that if the person takes more services, they will not churn
# as people get older, churn goes down


# In[ ]:




