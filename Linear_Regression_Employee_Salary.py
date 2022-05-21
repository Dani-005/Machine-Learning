#!/usr/bin/env python
# coding: utf-8

# Linear Regression: GOAL predict salary based on age, education level, sex

# dataset kaggle thank you for this dataset - https://www.kaggle.com/datasets/yasserh/employee-salaries-datatset

# In[139]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[222]:


salary = pd.read_csv('Downloads/salary.csv')
salary.head()


# In[223]:


salary.describe()


# In[224]:


salary.info()


# In[226]:


plt.scatter(salary['Gender'], salary['Salary'])


# In[227]:


plt.scatter(salary['Age'], salary['Salary'])


# In[228]:


# better for linear regression


# In[229]:


plt.scatter(salary['PhD'], salary['Salary'])


# In[230]:


X = salary[['Age']]
y = salary['Salary']


# In[231]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=1)


# In[232]:


clf=LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)


# In[233]:


y_pred = clf.predict(X_test)
print(y_pred)


# In[234]:


score = clf.score(X_test, y_test)
print(score)


# In[238]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax.scatter(X_train, y_train, color='blue')
ax.plot(X_test, y_pred, color='red', linewidth=1)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Salary', fontsize=12)
ax.set_title('Linear Regression Employee Salary')
fig.tight_layout()


# In[236]:


# not a great dataset for linear regression


# In[ ]:




