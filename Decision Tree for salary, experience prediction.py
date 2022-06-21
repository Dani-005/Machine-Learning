#!/usr/bin/env python
# coding: utf-8

# Decision Tree Machine Learning - Years of experience on the job, gender, & education level to determine salary

# Dataset from Kaggle:  https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists 
# combined with https://www.kaggle.com/datasets/rsadiq/salary

# Goal:  Predict salary of data scientist based on experience, gender, and education level. 

# In[275]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# In[276]:


wages = pd.read_csv('Documents/yrs_exp_salary.csv')
wages.head()


# In[277]:


wages.info()


# In[278]:


wages.describe()


# In[279]:


wages.isna().sum()


# In[280]:


wages1=wages.dropna(subset=['gender','education_level','major_discipline','company_type', 'last_new_job' ])


# In[281]:


wages1.isna().sum()


# In[282]:


inputs = wages1[['gender', 'education_level', 'experience']]
target = wages1['Salary']


# In[283]:


le_gender = LabelEncoder()
le_education_level = LabelEncoder()


# In[284]:


inputs['gender_n'] = le_gender.fit_transform(inputs['gender'])
inputs['education_level_n'] = le_education_level.fit_transform(inputs['education_level'])
inputs.head()


# In[285]:


list(inputs.gender_n)


# In[286]:


list(inputs.education_level_n)


# In[287]:


# drop the label columsn and create a new dataframe
inputs_n = inputs.drop(['gender', 'education_level'], axis='columns')
inputs_n
# gender_n 0:female, 1:male, education_level: 0:Graduate, 1:Masters


# In[293]:


X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.3, random_state=1)


# In[294]:


# train classifier
model = DecisionTreeClassifier()


# In[297]:


model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


# could hypertune this model to get better accuracy score


# In[298]:


model2 = DecisionTreeClassifier(criterion='entropy' )


# In[299]:


model2.fit(X_train, y_train)
model2.score(X_test, y_test)


# In[302]:


model3 = DecisionTreeClassifier(criterion='gini', splitter='random')


# In[303]:


model3.fit(X_train, y_train)
model3.score(X_test, y_test)


# In[325]:


model4 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=1, class_weight="balanced" )


# In[326]:


model4.fit(X_train, y_train)
model4.score(X_test, y_test)


# In[327]:


# Tried a number of parameters to change the score from 75. I did get 0 so went back to model4


# In[328]:


# run predictions, 5 yrs experience, male, masters
model4.predict([[5, 1, 1]])
# predicts salary of $93,940


# In[330]:


# predict 10 years experience, woman, graduate
model.predict([[10, 0, 0]])
# predicts salary $135,675


# In[331]:


# predict 10 years experience, woman, graduate
model4.predict([[10, 0, 0]])
# predicts salary $66,029
# not sure which is the better model, model vs model4 salary $135,675 vs $66,029


# In[ ]:




