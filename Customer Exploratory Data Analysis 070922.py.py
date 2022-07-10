#!/usr/bin/env python
# coding: utf-8

# Customer Data Analysis

# Explore Customer Data to Find Trends and Analysis.

# Thank you to Kaggle dataset: https://www.kaggle.com/datasets/mountboy/online-store-customer-data

# In[54]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import datetime as dt


# In[5]:


df = pd.read_csv('Documents/online_store_customer_data.csv')
df.head()


# In[6]:


df.shape


# In[7]:


df.describe() .T


# In[8]:


# remove all missing data rows because I am unable to figure out the best choice ie for age use mean, median? 
df.dropna(how='all', inplace=True)


# In[9]:


df.isna().sum()


# In[10]:


df.describe() .T


# In[11]:


df.rename(columns={'State_names' : 'State'}, inplace=True)


# In[12]:


df.dropna(inplace=True)


# In[13]:


df.isna().sum()


# In[14]:


df[['day', 'month', 'year']] = df['Transaction_date'].str.split('/', expand = True)


# In[15]:


df.head()


# In[69]:


df.tail()


# In[16]:


df.year.unique()


# In[17]:


df['Sex'] = df.Gender.map({'Female':0, 'Male':1, 'unknown':2})
df.head()


# In[18]:


df.Marital_status.unique()


# In[19]:


# change Marital_status column to numerical using .map, Married=0, Single=1
df['m0_s1'] = df.Marital_status.map({'Married':0, 'Single':1})
df.head()


# In[20]:


df.Employees_status.unique()


# In[21]:


# change Employees_status column to numerical using .map, Employees=0, self-employed=1, Unemployment=2, workers=3
df['employment_status'] = df.Employees_status.map({'Employees':0, 'self-employed':1, 'Unemployment':2, 'workers':3})
df.head()


# In[22]:


df.Payment_method.unique()


# In[23]:


# change Payment_method column to numerical using .map, Card=0, Other=1, PayPal=2
df['Payment'] = df.Payment_method.map({'Card':0, 'Other':1, 'PayPal':2})
df.head()


# In[41]:


df.groupby('year')['Amount_spent'].sum()


# In[25]:


df.groupby('year')['Amount_spent'].sum().plot.bar(color='blueviolet')
plt.xlabel('year')
plt.ylabel('Amount spent')
plt.title('Spend by year')
plt.show()


# In[26]:


medians = df.groupby('day')['Amount_spent'].median()
medians


# In[27]:


df.groupby('Gender')['Amount_spent'].sum().plot.bar(color='teal')
plt.xlabel('Gender')
plt.ylabel('Amount spent')
plt.title('Spend by Gender')
plt.show()


# In[28]:


df.groupby('Age')['Amount_spent'].sum().plot(color='violet')
plt.xlabel('Age')
plt.ylabel('Amount spent')
plt.title('Spend by Age')
plt.show()


# In[29]:


df.groupby('Segment')['Amount_spent'].sum().plot.bar(color='salmon')
plt.xlabel('Segment')
plt.ylabel('Amount spent')
plt.title('Spend by Segment')
plt.show()


# In[30]:


df.groupby('Employees_status')['Amount_spent'].sum().plot.bar(color='green')
plt.xlabel('Employees_status')
plt.ylabel('Amount spent')
plt.title('Spend by Employees_status')
plt.show()


# In[31]:


df.groupby('Payment_method')['Amount_spent'].sum().plot.bar(color='dodgerblue')
plt.xlabel('Payment_method')
plt.ylabel('Amount spent')
plt.title('Spend by Payment_method')
plt.show()


# In[64]:


plt.figure(figsize=(10,5))
df.groupby('State')['Amount_spent'].sum().sort_values(ascending=False).head(10).plot.bar(color='palegreen')
plt.ylabel('Amount_spent')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.title('Top 10 States in terms of amount_spent')


# In[70]:


plt.figure(figsize=(10,5))
df.groupby('State')['Amount_spent'].sum().sort_values(ascending=True).head(10).plot.bar(color='mediumseagreen')
plt.ylabel('Amount_spent')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.title('Top 10 States in terms of least amount_spent')


# In[33]:


df["Transaction_date"] = pd.to_datetime(df["Transaction_date"], dayfirst = True).dt.date


# In[66]:


plt.figure(figsize=(20,5))
df.groupby('Transaction_date')['Amount_spent'].sum().plot(color='violet')
plt.title('Spend by date')
plt.xlabel('Transaction date')
plt.ylabel('Amount spent')
plt.show()


# In[68]:


df.to_pickle("070922.pickle")


# Analysis:
# 
# Amount spent over 3 year period, 2019, 2020, 2021
# 
# Most spent in 2019, least 2021
# 
# More women spent money than men
# 
# Certain ages 50 or above were higher spenders, lower spenders were under 50
# 
# The basic segment was purchased most frequently
# 
# Most spend was by employees
# 
# Paypal used most as way of paying
# 
# Massachusetts, Arizona, and Illinois were top 3 states with most spend vs Alabama, South Carolina, and Kansas with least amount spent

# Thank you for taking the time to view my analysis

# In[ ]:




