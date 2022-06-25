#!/usr/bin/env python
# coding: utf-8

# Support Vector Machines (SVM) Machine Learning - Using SVM, PCA, DownSampling

# Dataset from Kaggle:  https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset      
# Code From Youtube Video StatQuest  https://www.youtube.com/watch?v=8A7L0GsBiLQ

# Goal:  Predict if a credit card user will default 

# In[72]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


# In[21]:


df = pd.read_csv('Documents/UCI_Credit_Card.csv')
df.head()


# This is the detail regarding the features:  
# 
# ID: ID of each client
# 
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# 
# SEX: Gender (1=male, 2=female)
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# AGE: Age in years
# 
# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# PAY_2: Repayment status in August, 2005 (scale same as above)
# PAY_3: Repayment status in July, 2005 (scale same as above)
# PAY_4: Repayment status in June, 2005 (scale same as above)
# PAY_5: Repayment status in May, 2005 (scale same as above)
# PAY_6: Repayment status in April, 2005 (scale same as above)
# 
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# 
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# 
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# 
# default.payment.next.month: Default payment (1=yes, 0=no)

# In[22]:


df.columns


# In[23]:


# change 'default.payment.next.month' to 'default'
df.rename({'default.payment.next.month' : 'default'}, axis='columns', inplace=True)
df.head()


# In[24]:


# 'ID' not necessary so drop
df.drop('ID', axis=1, inplace=True)
df.head()


# In[25]:


# look for missing data
df.describe()


# In[26]:


df.info()


# In[27]:


df.isna().sum()


# In[28]:


df.shape


# In[29]:


# check for values for certain attributes
df.SEX.unique()
# good 1:male  2:female


# In[30]:


df.EDUCATION.unique()
# 0 may contain missing values


# In[31]:


df.MARRIAGE.unique()
# 0 may contain missing values


# In[38]:


# find how many rows contain the missing value 0
df.EDUCATION.value_counts()


# In[39]:


df.MARRIAGE.value_counts()


# In[43]:


df_no_missing = df.loc[((df['EDUCATION']!=0) & (df['MARRIAGE']!=0))]


# In[46]:


len(df_no_missing)


# In[47]:


df_no_missing['EDUCATION'].unique()


# In[48]:


df_no_missing['MARRIAGE'].unique()


# In[49]:


df_no_default = df_no_missing[df_no_missing['default']==0]
df_default = df_no_missing[df_no_missing['default']==1]


# In[50]:


# data too large so downsample to 1000 for both default and no default
df_no_default_downsampled = resample(df_no_default,
                                    replace=False,
                                    n_samples=1000,
                                    random_state=1)
len(df_no_default_downsampled)


# In[56]:


df_default_downsampled = resample(df_default,
                                 replace=False,
                                 n_samples=1000,
                                 random_state=1)

len(df_default_downsampled)


# In[60]:


df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])
len(df_downsample)


# In[62]:


# format data
X = df_downsample.drop('default', axis=1).copy()
X.head()


# In[64]:


y = df_downsample['default'].copy()
y.head()


# In[66]:


# OneHotEncode data so that categorical data: sex education marriage pay
# are binary numerical not categorical for SVM
X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE',
                                      'PAY_0','PAY_2','PAY_3',
                                      'PAY_4','PAY_5','PAY_6',])
X_encoded.head()


# In[67]:


# center and scale, to balance the data each column should have a mean of 0 and STD of 1
X_train, X_test, y_train, y_test=train_test_split(X_encoded, y, test_size=0.3, random_state=1)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)


# In[69]:


clf_svm = SVC(random_state=1)
clf_svm.fit(X_train_scaled, y_train)


# In[70]:


plot_confusion_matrix(clf_svm, X_test_scaled, y_test, 
                      values_format='d', 
                      display_labels=['Did not default', 'Defaulted'])


# In[74]:


# try to improve model optimizing parameters using cross validation
# and grid search cv
param_grid= [
    {'C': [0.5,1,10,100], 
    'gamma' : ['scale', 1, 0.1, 0.001, 0.0001],
    'kernel': ['rbf']}
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0)
# there are other params that could be hypertuned

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)


# In[75]:


# create model based on optimal params
clf_svm = SVC(C=1, gamma=0.001, random_state=1)
clf_svm.fit(X_train, y_train)


# In[76]:


plot_confusion_matrix(clf_svm, X_test_scaled, y_test, 
                      values_format='d',
                     display_labels=['Did not default','Defaulted'])


# In[77]:


# draw decision boundary, how many columns are in dataset?
len(df_downsample.columns)


# In[78]:


# use PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.xlabel('Principal Components')
plt.ylabel('Percentage of explained variance')
plt.title('Scree Plot')
plt.show()
# the goal is to have the first 2 columns on left to be tallest,
# first column good, second column not good


# In[ ]:


# Continue on with PCA of 2 components, youtube: 41:35


# In[ ]:


# however, prediction of future data not predicted


# In[ ]:




