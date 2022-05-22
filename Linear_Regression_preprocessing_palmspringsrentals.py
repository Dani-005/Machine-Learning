#!/usr/bin/env python
# coding: utf-8

# Linear Regression preprocessing: GOAL rent prices based on features, dataset issues, not good for LR

# dataset kaggle thank you for this dataset - 

# In[426]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


# In[427]:


rent = pd.read_csv('Documents/palm_springs_rental.csv')
rent.head()


# In[428]:


rent.describe()


# In[429]:


print(rent.groupby('numPeople').sum())


# In[430]:


rent['numPeople']= rent['numPeople'].replace(['16+'], ['16'])


# In[431]:


print(rent.groupby('numPeople').sum())


# In[432]:





# In[434]:





# In[436]:


rent['upto_people'] = rent['numPeople'].replace(['6','7','8','9','10'], ['10','10','10','10','10'])


# In[437]:


rent['upto_people'] = rent['numPeople'].replace(['11','12','13','14','15'], ['15','15','15','15','15'])


# In[438]:


rent['upto_people'] = rent['numPeople'].replace(['16','18','20','26'], ['15','15','15','15'])


# In[439]:


rent.head()


# In[440]:


rent.info()


# In[443]:





# In[444]:


rent.isnull().sum()


# In[445]:


rent.drop(['numBed','prices.period'], axis=1, inplace=True)


# In[446]:


rent.info()


# In[448]:


rent.isnull().sum()


# In[449]:


rent=rent.dropna(how='any')


# In[450]:


rent.isnull().sum()


# In[451]:


rent.describe()


# In[453]:


rent[89:90]


# In[454]:


rent = rent.astype({'numBathroom':'int', 'numPeople':'int'})
#rent = rent.astype({'numPeople':'int', 'prices':'float', 'minstay':'int', 'upto_people':'int'})


# In[455]:


rent.info()


# In[466]:


#rent['prices']=rent['prices'].astype('int32')


# In[467]:


print(rent.groupby('prices').sum())


# In[472]:


print(rent.loc[(rent['prices']== 'D 93') ])


# In[474]:


print(rent.drop([rent.index[5145]]))


# In[481]:


print(rent.loc[(rent['prices']== 'D 93') ])


# In[475]:


print(rent.groupby('prices').sum())


# In[479]:


print(rent.drop(rent[rent['prices']=='D93'].index))


# In[477]:


print(rent.groupby('prices').sum())


# In[482]:


rent = rent.drop(rent.index[rent['prices']== 'D 93'])


# In[483]:


print(rent.groupby('prices').sum())


# In[484]:


rent = rent.drop(rent.index[rent['prices']== 'D 90'])
rent = rent.drop(rent.index[rent['prices']== 'D 94'])
rent = rent.drop(rent.index[rent['prices']== 'D 95'])
rent = rent.drop(rent.index[rent['prices']== 'D 98'])
rent = rent.drop(rent.index[rent['prices']== 'D 99'])


# In[485]:


print(rent.groupby('prices').sum())


# In[ ]:





# In[ ]:





# In[ ]:




