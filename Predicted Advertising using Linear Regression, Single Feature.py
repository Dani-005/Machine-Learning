#!/usr/bin/env python
# coding: utf-8

# Linear Regression Machine Learning - Restaurant TV Advertising Prediction

# Dataset from Kaggle:  https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/data

# Goal:  What is the prediction of TV advertising affecting sales

# In[27]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


adv = pd.read_csv('Downloads/advertising.csv')
adv.head()


# In[4]:


adv.describe()


# In[5]:


adv.info()


# In[6]:


plt.scatter(adv.TV, adv.Sales, color='blue', marker='^')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')


# In[7]:


X = adv[['TV']]
y = adv['Sales']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# In[12]:


model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# result is a 76% accuracy in the machine learning model


# Run predictions: 

# If spent $125 per week on TV advertising, what is predicted sales?

# If spent $325 per week on TV advertising, what is the predicted sales?

# In[15]:


model.predict([[125]])
# $13.89 increase in sales


# In[16]:


model.predict([[325]])
# 25.44 increase in sales


# In[47]:


# feed in a list of different TV advertising price points to get out a list of predicted sales

budget = pd.read_csv('Documents/TV_budget_lr.csv')
budget.head()


# In[53]:


xt = budget[:6]

pred1 = model.predict(xt)
#model.predict(np.array([[50,100]]))


# In[54]:


pred1


# In[59]:


budget['Predicted_sales'] = pred1


# In[60]:


budget


# In[ ]:




