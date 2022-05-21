#!/usr/bin/env python
# coding: utf-8

# Linear Regression: GOAL predict sales based on advertising types

# dataset kaggle thank you for this dataset - https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data

# In[139]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[240]:


mkt_sales = pd.read_csv('Downloads/Dummy Data HSS.csv')
mkt_sales.head()


# In[241]:


mkt_sales.describe()


# In[242]:


# look for rows with missing data
mkt_sales.info()


# In[243]:


mkt_sales.isna().sum()


# In[244]:


# remove rows with missing data
mkt_sales = mkt_sales.dropna(how='any')


# In[245]:


mkt_sales.isna().sum()


# In[246]:


mkt_sales.shape


# In[247]:


plt.scatter(mkt_sales['TV'], mkt_sales['Sales']) #great dataset for linear regression


# In[248]:


plt.scatter(mkt_sales['Radio'], mkt_sales['Sales']) 


# In[249]:


plt.scatter(mkt_sales['Social Media'], mkt_sales['Sales'])


# In[250]:


plt.scatter(mkt_sales['Influencer'], mkt_sales['Sales'])


# In[251]:


mkt_sales.columns


# In[252]:


X = mkt_sales[['TV', 'Radio', 'Social Media']]
y = mkt_sales['Sales']            


# In[253]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)


# In[254]:


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[266]:


y_pred = clf.predict(X_test)


# In[267]:


print(y_pred)


# In[268]:


print(clf.score(X_test, y_test))


# In[258]:


#great score, great dataset for Linear Regression


# In[ ]:




