#!/usr/bin/env python
# coding: utf-8

# Linear Regression: GOAL Predict sales of ice cream flavors

# dataset kaggle thank you for this dataset - https://www.kaggle.com/datasets/sunlight1/icecream-shop-analysis 

# In[554]:


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


# In[555]:


treat = pd.read_csv('Documents/icecream_flavors.csv')
treat.head()


# In[556]:


treat.describe()


# In[557]:


treat.info()


# In[559]:


# I removed 4 values for vanilla icecream weeks 1-4, adding NaN in those fields in the csv doc
treat.isnull().sum()


# In[608]:


treat.groupby('week').sum()


# In[566]:


print(treat.isnull().any(axis=1))


# In[567]:


treat_find_null = treat.isnull().any(axis=1)
treat[treat_find_null]


# In[579]:


treat.at[3, 'week'] =1  


# In[580]:


treat_find_null = treat.isnull().any(axis=1)
treat[treat_find_null]


# In[581]:


treat.at[7, 'week'] =2 


# In[582]:


treat_find_null = treat.isnull().any(axis=1)
treat[treat_find_null]


# In[583]:


treat.at[11, 'week'] =3
treat.at[15, 'week'] =4


# In[585]:


treat.head(20)


# In[586]:


treat.groupby('flavor').sum()


# In[587]:


plt.scatter(treat['flavor'], treat['units sold'] )


# In[588]:


flavor_cat = treat['flavor']
label_encoder = LabelEncoder()
flavor_encoded = label_encoder.fit_transform(flavor_cat)
flavor_encoded[0:25]


# In[590]:


binary_encoder = OneHotEncoder(categories='auto')
flavor_1hot = binary_encoder.fit_transform(flavor_encoded.reshape(-1,1))
flavor_1hot_mat = flavor_1hot.toarray()
flavor_DF = pd.DataFrame(flavor_1hot_mat, columns=['chocolate', 'lemon', 'strawberry', 'vanilla' ])
flavor_DF.head()


# In[591]:


treat =  pd.concat([treat, flavor_DF], axis=1, verify_integrity=True)
treat.head()


# In[592]:


treat.columns


# In[593]:


X = treat[['chocolate', 'lemon', 'strawberry','vanilla']]
y = treat['units sold']


# In[594]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)


# In[595]:


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[596]:


clf.predict(X_test)


# In[598]:


clf.score(X_test, y_test)


# In[599]:


X = treat[['week','lemon']]
y = treat['units sold']


# In[600]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)


# In[601]:


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[ ]:


clf.predict(X_test)


# In[602]:


clf.score(X_test, y_test)


# In[603]:


X = treat[['week']]
y = treat['units sold']


# In[604]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)


# In[605]:


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[606]:


clf.predict(X_test)


# In[607]:


clf.score(X_test, y_test)


# In[ ]:


# I think you need to change your variables instead of using the same variables. 
 #I believe that is why the score is negative. 


# In[ ]:




