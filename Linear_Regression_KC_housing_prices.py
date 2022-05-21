#!/usr/bin/env python
# coding: utf-8

# Linear Regression: goal predict price based on area

# 

# dataset kaggle - KC Housing data

# In[139]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[190]:


gen_hou = pd.read_csv('Downloads/gen_housing.csv')
gen_hou.head()


# In[191]:


gen_hou.describe()


# In[192]:


gen_hou.info()


# In[193]:


# no missing values


# In[196]:


plt.scatter(gen_hou['area'], gen_hou['price'])


# In[197]:


plt.scatter(gen_hou['parking'], gen_hou['price'])


# In[198]:


plt.scatter(gen_hou['stories'], gen_hou['price'])


# In[199]:


plt.scatter(gen_hou['bedrooms'], gen_hou['price'])


# In[200]:


plt.scatter(gen_hou['bathrooms'], gen_hou['price'])


# In[201]:


#only linear regression is with area to predict
X = gen_hou[['area']]
y = gen_hou['price']


# In[204]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


# In[206]:


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[213]:


y_pred = (clf.predict(X_test))


# In[214]:


clf.score(X_test, y_test)


# In[210]:


# not a good score, not a very good dataset to predict prices. 


# In[221]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax.scatter(X_train,y_train, color='blue')
ax.plot(X_test, y_pred, color='red', linewidth = 1)
ax.tick_params(labelsize=12)
ax.set_xlabel('area', fontsize=12)
ax.set_ylabel('price', fontsize=12)
ax.set_title('Area vs price', fontsize=14)
fig.tight_layout()


# In[ ]:




