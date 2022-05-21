#!/usr/bin/env python
# coding: utf-8

# Linear Regression Goal: Predict house prices

# Thank you for the dataset found on Kaggle: https://www.kaggle.com/datasets/rishabhkatoch/house-price

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
sb.set_style('whitegrid')
rcParams['figure.figsize']= 5,4


# In[11]:


prices = pd.read_csv('Documents/House_Prices.csv')
prices.head()


# In[12]:


prices.shape


# In[13]:


prices.info()


# In[14]:


prices.describe()


# In[15]:


sb.pairplot(prices)


# In[16]:


print(prices.corr())


# In[17]:


prices_data = prices[['SqFt','Bedrooms','Bathrooms']].values
prices_target = prices[['Price']].values

prices_data_names = ['SqFt','Bedrooms','Bathrooms']
X,y = scale(prices_data), prices_target


# In[20]:


linreg = LinearRegression(normalize=True)
linreg.fit(X,y)
print(linreg.score(X,y))


# In[21]:


prices2 = pd.read_csv('Documents/House_Prices.csv')
prices2.head()


# In[22]:


X = prices[['SqFt','Bedrooms','Bathrooms']].values
y = prices[['Price']].values


# In[23]:


linreg = LinearRegression(normalize=True)
linreg.fit(X,y)
print(linreg.score(X,y))


# As you can see in the pairplot, there is not a lot of linear regression with this data.  This model has a R squared value of .43 

# Based on the SqFt, Bedrooms, Bathrooms, the price does not follow a linear regression 
