#!/usr/bin/env python
# coding: utf-8

# Logistic Regression to predict specific absenteeism occuring at work

# Thank you for the data set from Kaggle:  https://www.kaggle.com/datasets/joelpires/absenteeism-at-works

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


abs = pd.read_csv('Downloads/absenteeism.csv')
abs.head()


# In[11]:


abs.isna().sum()
# no missing data


# In[12]:


abs.info()
# all columns are numeric 


# In[8]:


abs.describe()


# In[9]:


abs1 = abs


# In[10]:


# copy of dataframe  to work on
abs1.head()


# In[14]:


month_cat = abs1['Month of absence']
label_encoder = LabelEncoder()
month_encoded = label_encoder.fit_transform(month_cat)
month_encoded[0:40]


# In[18]:


# sunday = 1
day_cat = abs1['Day of the week']
label_encoder = LabelEncoder()
day_encoded = label_encoder.fit_transform(day_cat)
day_encoded[0:40]


# In[16]:


# summer 1, fall 2, winter 3, spring 4
seasons_cat = abs1['Seasons']
label_encoder = LabelEncoder()
seasons_encoded = label_encoder.fit_transform(seasons_cat)
seasons_encoded[0:40]


# In[23]:


abs2 = abs


# In[24]:


abs2.head()


# In[31]:


abs1['Month of absence'].value_counts()


# In[35]:


# onehotencode data
binary_encoder = OneHotEncoder(categories='auto')
month_1hot = binary_encoder.fit_transform(month_encoded.reshape(-1,1))
month_1hot_mat = month_1hot.toarray()
month_DF = pd.DataFrame(month_1hot_mat, columns=['0','1','2','3','4','5','6','7','8','9','10','11','12'])


# In[36]:


month_DF['1'].value_counts()


# In[37]:


print(month_1hot_mat)


# In[39]:


abs1['Day of the week'].value_counts()
# work days tracked is only Monday - Friday, not weekends, sunday=1, saturday=7


# In[40]:


# first time error, go back and value_count the columns same as months issue
day_1hot = binary_encoder.fit_transform(day_encoded.reshape(-1,1))
day_1hot_mat = day_1hot.toarray()
day_DF = pd.DataFrame(day_1hot_mat, columns=['2','3','4','5','6'])
day_DF.head()


# In[33]:


seasons_1hot = binary_encoder.fit_transform(seasons_encoded.reshape(-1,1))
seasons_1hot_mat = seasons_1hot.toarray()
seasons_DF = pd.DataFrame(seasons_1hot_mat, columns=['summer', 'fall','winter', 'spring'])
seasons_DF.head()


# In[42]:


# remove non necessary columns
abs1.columns


# In[46]:


#abs1.drop(['Month of absence', 'Day of the week','Seasons'], axis=1, inplace=True)


# In[48]:


# concat the dataframes together
abs_alys = pd.concat([abs1, month_DF, seasons_DF], axis=1, verify_integrity=True)


# In[50]:


# can you predict when employees may be absent


# In[ ]:


X = abs_alys[['Age']]
y = abs_alys[['summer']]


# In[51]:


# preduct age of employee and if they are absent in the summer
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)


# In[52]:


model = LogisticRegression()


# In[53]:


model.fit(X_train, y_train)


# In[54]:


X_test


# In[55]:


y_predicted = model.predict(X_test)


# In[57]:


y_predicted = model.predict_proba(X_test)


# In[58]:


model.score(X_test, y_test)


# In[ ]:





# In[62]:


X_dist_sum = abs_alys[['Distance from Residence to Work']]
y_dist_sum = abs_alys[['summer']]


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X_dist_sum,y_dist_sum,test_size=0.2, random_state=1)


# In[64]:


model_dist_sum = LogisticRegression()


# In[65]:


model_dist_sum.fit(X_train, y_train)


# In[66]:


y_predicted_dist_sum = model_dist_sum.predict(X_test)


# In[67]:


model_dist_sum.score(X_test, y_test)


# In[ ]:


# same score as age and missing work in the summer


# In[72]:


plt.scatter(abs_alys['Distance from Residence to Work'], abs_alys['winter'])


# In[73]:


# is there a higher incidence with winter living further away from work and absenteeism


# In[74]:


X_dist_wnt = abs_alys[['Distance from Residence to Work']]
y_dist_wnt = abs_alys[['winter']]


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X_dist_wnt,y_dist_wnt,test_size=0.2, random_state=1)


# In[76]:


model_dist_wnt = LogisticRegression()


# In[77]:


model_dist_wnt.fit(X_train, y_train)


# In[78]:


y_predicted_dist_wnt = model_dist_wnt.predict(X_test)


# In[79]:


model_dist_wnt.score(X_test, y_test)


# In[ ]:


# model has accuracy of 76 for missing work in winter based on distance of the residence from work


# In[ ]:




