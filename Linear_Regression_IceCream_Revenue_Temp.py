#!/usr/bin/env python
# coding: utf-8

# Linear Regression Goal: Predict icecream sales based on temperature 

# Thank you for the dataset found on Kaggle: https://www.kaggle.com/datasets/vinicius150987/ice-cream-revenue

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
sb.set_style('whitegrid')
rcParams['figure.figsize'] = 5,4


# In[31]:


rev = pd.read_csv('Documents/IceCreamData.csv')
rev.head()


# In[32]:


rev.info()


# In[34]:


rev.shape


# In[35]:


rev.describe()


# In[43]:


X = rev.loc[:,['Temperature']].values 
y = rev.loc[:,['Revenue']].values


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=21)


# In[45]:


X_train.shape


# In[47]:


X_test.shape


# In[48]:


reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)


# In[55]:


#check score
score = reg.score(X_test, y_test)
print(score)


# In[56]:


y_pred = reg.predict(X_test)


# In[57]:


print('Coefficients: \n', reg.coef_)
print('Mean squared error: %.2f'%mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' %r2_score(y_test, y_pred))


# In[74]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax.scatter(X_train,y_train,color='black')
ax.plot(X_test, y_pred, color='blue', linewidth=1)
ax.set_xlabel('Temperature (C)', fontsize=12)
ax.set_ylabel('Ice Cream Revenue', fontsize=12)
ax.set_title('Linear Regression Ice Cream Sales based on Temperature', fontsize=12)
fig.tight_layout()


# Analysis: As temperature increases, ice cream sales are predicted to increase. R squared score is .98 which is a strong model 
