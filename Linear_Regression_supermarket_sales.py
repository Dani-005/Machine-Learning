#!/usr/bin/env python
# coding: utf-8

# Linear Regression: GOAL predict sales based on supermarket features

# dataset kaggle thank you for this dataset - https://www.kaggle.com/datasets/surajjha101/stores-area-and-sales-data

# In[278]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[270]:


store = pd.read_csv('Downloads/stores.csv')
store.head()


# In[271]:


store.describe()


# In[272]:


store.info()


# In[274]:


plt.scatter(store['Store_Area'], store['Store_Sales'])


# In[275]:


plt.scatter(store['Items_Available'], store['Store_Sales'])


# In[276]:


plt.scatter(store['Daily_Customer_Count'], store['Store_Sales'])


# In[279]:


sb.pairplot(store)


# In[280]:


X = store[['Store_Area', 'Items_Available', 'Daily_Customer_Count']]
y = store['Store_Sales']


# In[281]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
clf.fit(X_train, y_train)


# In[282]:


y_pred = clf.predict(X_test)


# In[283]:


clf.score(X_test, y_test)


# In[284]:


#no linear regression with this dataset


# In[285]:


print(store.corr())


# In[286]:


#possibly my datasets are too small


# In[ ]:




