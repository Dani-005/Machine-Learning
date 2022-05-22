#!/usr/bin/env python
# coding: utf-8

# Linear Regression preprocessing: GOAL preprocess for used fiat linear regression

# dataset kaggle thank you for this dataset - https://www.kaggle.com/datasets/paolocons/small-dataset-about-used-fiat-500-sold-in-italy

# In[535]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report


# In[496]:


car = pd.read_csv('Downloads/Used_fiat.csv')
car.head()


# In[497]:


car.describe()


# In[498]:


car.info()


# In[533]:


sb.heatmap(car.corr())


# In[500]:


car.groupby('model').sum()


# In[503]:


def model_no(row_number, assigned_value):
    return assigned_value[row_number]
model_type = {'lounge': 1, 'pop': 2, 'sport': 3, 'star': 4}
car['model_code']= car['model'].apply(model_no, args=(model_type,))
car.head()


# In[504]:


car.groupby('transmission').sum()


# In[505]:


def trans_type(row_number, assigned_value):
    return assigned_value[row_number]
trans_no = {'automatic':1, 'manual':2}
car['trans_code'] = car['transmission'].apply(trans_type, args=(trans_no,))
car.head()


# In[506]:


# ready for Linear Regression
car.columns


# In[508]:


X = car[['model_code', 'trans_code', 'engine_power']]
y = car['price']


# In[509]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)


# In[510]:


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[511]:


y_pred = clf.predict(X_test)


# In[513]:


clf.score(X_test, y_test)


# In[519]:


A= car[['model_code', 'age_in_days' ]]
b = car['price']


# In[520]:


A_train, A_test, b_train, b_test = train_test_split(A,b, test_size=0.2, random_state=1)


# In[521]:


A_train.shape


# In[522]:


ma = LinearRegression(fit_intercept=True)
ma.fit(A_train, b_train)


# In[529]:


b_pred = ma.predict(A_test)


# In[530]:


ma.score(A_test, b_test)


# In[526]:


#better model to predict price based on type of car and age of car


# In[531]:


# evaluate model
print('Coefficients: \n', ma.coef_)
print('Mean squared error: %.2f' %mean_squared_error(b_test, b_pred ))
print('Coefficient of determination: %.2f'%r2_score(b_test, b_pred))


# In[ ]:





# In[ ]:




