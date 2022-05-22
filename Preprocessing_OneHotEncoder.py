#!/usr/bin/env python
# coding: utf-8

# Linear Regression preprocessing: GOAL OneHotEncoder convert region to 4 columns OneHotEncoded

# dataset kaggle thank you for this dataset - insurance

# In[539]:


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


# In[541]:


car = pd.read_csv('Documents/insurance.csv')
car.head()


# In[542]:


region_cat = car['region']
label_encoder = LabelEncoder()
region_encoded = label_encoder.fit_transform(region_cat)
region_encoded[0:100]


# In[543]:


binary_encoder=OneHotEncoder(categories='auto')
region_1hot = binary_encoder.fit_transform(region_encoded.reshape(-1,1))
region_1hot_mat = region_1hot.toarray()
region_DF = pd.DataFrame(region_1hot_mat, columns=['NE', 'NW', 'SE', 'SW'])
region_DF.head()


# In[544]:


# goal completed!


# In[ ]:




