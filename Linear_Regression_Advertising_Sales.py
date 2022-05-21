#!/usr/bin/env python
# coding: utf-8

# Linear Regression Goal: Predict Sales based on different types of advertising

# Thank you for the dataset found on Kaggle: https://www.kaggle.com/code/saikatkumardey/linear-regression-case-study/data

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


# In[75]:


adv = pd.read_csv('Documents/adv_sales.csv')
adv.head()


# In[76]:


adv.columns


# In[77]:


adv.info()


# In[78]:


adv.describe()


# In[79]:


#complete dataset no nulls, use columns: tv, radio, newspaper, area_suburban, area_urban to predict sales


# In[80]:


adv_data = adv[['TV', 'radio', 'newspaper', 'Area_suburban', 'Area_urban']].values
adv_target = adv[['sales']].values


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(adv_data, adv_target, test_size=0.3, random_state=1)


# In[82]:


linreg = LinearRegression(fit_intercept=True)
linreg.fit(X_train, y_train)


# In[87]:


score = linreg.score(X_train, y_train)
print(score)


# In[88]:


y_pred = linreg.predict(X_test)


# In[91]:


print('Coefficients: \n', linreg.coef_) 
print('Mean squared error: %.2f' %mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' %r2_score(y_test, y_pred))


# MSE of 1.92 is pretty good, closer to 0 is best, R squared is pretty good at .92 where I wanted to be as close to 1 as possible. 

# In[93]:


# go back and analyse each feature
adv_sales = pd.read_csv('Documents/adv_sales.csv')
adv_sales.head()


# In[94]:


sb.pairplot(adv_sales)


# In[95]:


# look at the correlation between features
print(adv_sales.corr())


# In[97]:


X = adv_sales[['TV']].values
y = adv_sales[['sales']].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


# In[98]:


tv_reg=LinearRegression(fit_intercept=True)
tv_reg.fit(X_train, y_train)


# In[99]:


tv_pred = tv_reg.predict(X_test)


# In[103]:


score_test = tv_reg.score(X_test, y_test)
print(score)


# In[104]:


score_train = tv_reg.score(X_train, y_train)
print(score_train)


# In[105]:


#use one 1 feature - x=radio and y=sales
#load dataset
adv = pd.read_csv('Documents/adv_sales.csv')
adv.head()


# In[106]:


X = adv.loc[:,['radio']].values
X.shape


# In[107]:


y=adv.loc[:,'sales'].values
y.shape


# In[108]:


reg = LinearRegression(fit_intercept=True)


# In[110]:


reg.fit(X,y)


# In[115]:


reg_pred=reg.predict(X[0].reshape(-1,1))


# In[116]:


score=reg.score(X,y)
print(score)


# In[117]:


reg.coef_


# In[118]:


reg.intercept_


# In[123]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax.scatter(X,y, color='black')
ax.plot(X, reg.predict(X), color='red', linewidth=1)
ax.grid(True, axis='both', zorder=0, linestyle='--', color='k')
ax.tick_params(labelsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Linear Regression radio sales')
fig.tight_layout()


# In[125]:


#create model for newspaper advertising to predict sales
news_adv = pd.read_csv('Documents/adv_sales.csv')
news_adv.head()


# In[127]:


X=news_adv.loc[:,['newspaper']].values
y=news_adv.loc[:,'sales'].values
X.shape


# In[128]:


y.shape


# In[129]:


reg=LinearRegression(fit_intercept=True)
reg.fit(X,y)


# In[131]:


y_pred = reg.predict(X)


# In[132]:


print(y_pred)


# In[133]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax.scatter(X,y, color='black')
ax.plot(X, y_pred, color='red', linewidth=1)
ax.grid(True, axis='both', zorder=0, linestyle='--', color='k')
ax.tick_params(labelsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Linear Regression newspaper sales')
fig.tight_layout()


# In[ ]:




