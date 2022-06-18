#!/usr/bin/env python
# coding: utf-8

# KMeans Machine Learning - Find potential customers who spend more at the Mall based on Membership Program

# Dataset from Kaggle:  https://www.kaggle.com/datasets/kandij/mall-customers

# Goal:  Find groups of customers who have higher potential to spend more money at a mall restaurant

# In[189]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# In[181]:


mall_cust = pd.read_csv('Documents/Mall_Customers.csv')
mall_cust.head()


# In[182]:


mall_cust.describe()


# In[183]:


mall_cust.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)


# In[184]:


mall_cust.describe()


# In[185]:


mall_cust.isna().sum()


# In[186]:


mall_cust['Sex'] = mall_cust.Genre.map({'Female':0, 'Male':1})
mall_cust.head()


# In[187]:


mall_cust.drop('Genre', axis=1, inplace=True)


# In[188]:


mall_cust.head()


# In[244]:


# did not split data between train and test for this analysis


# Are there any groups of customers who have a higer spending score based on age or income? 

# In[190]:


# K Means cluster Age - Income
age = mall_cust


# In[192]:


# find KMeans elbow for age
age_range = range(1,10)
age_sse = []
for k in age_range:
    km = KMeans(n_clusters=k)
    km.fit(age[['Age','Score']])
    age_sse.append(km.inertia_)


# In[193]:


plt.xlabel('K - Age')
plt.ylabel('Sum of squared error')
plt.plot(age_range, age_sse)
# result shows 4 age n_clusters 


# In[200]:


income = mall_cust


# In[201]:


# find KMeans elbow for income
income_range = range(1,10)
income_sse = []
for k in income_range:
    km = KMeans(n_clusters=k)
    km.fit(income[['Income','Score']])
    income_sse.append(km.inertia_)


# In[202]:


plt.xlabel('K - Income')
plt.ylabel('Sum of squared error')
plt.plot(income_range, income_sse)
# result shows 5 income n_clusters 


# In[198]:


# find age clusters
km_age = KMeans(n_clusters=4)
km_age


# In[203]:


y_age_predicted = km_age.fit_predict(age[['Age', 'Score']])
y_age_predicted


# In[204]:


age['age_cluster'] = y_age_predicted
age.head(3)


# In[205]:


km_age.cluster_centers_


# In[214]:


age0 = age[age.age_cluster==0]
age1 = age[age.age_cluster==1]
age2 = age[age.age_cluster==2]
age3 = age[age.age_cluster==3]

plt.scatter(age0.Age, age0.Score, color='red')
plt.scatter(age1.Age, age1.Score, color='blue')
plt.scatter(age2.Age, age2.Score, color='green')
plt.scatter(age3.Age, age3.Score, color='yellow')

plt.scatter(km_age.cluster_centers_[:,0], km_age.cluster_centers_[:,1], color='black', marker='*', label='centroid')

plt.xlabel('Age')
plt.ylabel('Score')
plt.legend()


# Observation: Based on 4 Age clusters of shoppers in the Membership Program, you can target marketing to those who have a higher spending score. After age 60, those members have a lower spending score. 

# In[208]:


# find age clusters
km_income = KMeans(n_clusters=5)
km_income


# In[209]:


y_income_predicted = km_income.fit_predict(income[['Income', 'Score']])
y_income_predicted


# In[210]:


income['income_cluster'] = y_income_predicted
income.head(3)


# In[211]:


km_income.cluster_centers_


# In[213]:


income0 = income[income.income_cluster==0]
income1 = income[income.income_cluster==1]
income2 = income[income.income_cluster==2]
income3 = income[income.income_cluster==3]
income4 = income[income.income_cluster==4]


plt.scatter(income0.Income, income0.Score, color='red')
plt.scatter(income1.Income, income1.Score, color='blue')
plt.scatter(income2.Income, income2.Score, color='green')
plt.scatter(income3.Income, income3.Score, color='yellow')
plt.scatter(income4.Income, income4.Score, color='orange')


plt.scatter(km_income.cluster_centers_[:,0], km_income.cluster_centers_[:,1], color='black', marker='*', label='centroid')

plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()


# Observation: Based on 5 Income clusters of shoppers in the Membership Program.  Depending on your marketing campaign, there are 5 distinct groups of spenders by income. 

# Visualization of Data

# In[227]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sb.countplot(x='age_cluster', hue='Sex', data=age)
plt.title("Female(0) vs Male(1) Ratio in Age_Cluster")

plt.subplot(1,2,2)
sb.countplot(x='income_cluster', hue='Sex', data=income)
plt.title("Female(0) vs Male(1) Ratio in Income_Cluster")
plt.show()


# In[238]:


# perform predictions on the model, (Age 27, Spending score 75), (Age 30, Spending score 30),(Age 25, Spending score 15)
a1_new = [[27, 75], [30, 30], [25, 15]]
a1_age_predicted = km_age.predict(a1_new)
print(a1_age_predicted)
# Age cluster 0, 3, 1


# In[239]:


# perform predictions on the model:(age 55, spending score 62), (age 15, spending score 15)
a2_new = [[55, 62], [15,15]]
a2_age_predicted = km_age.predict(a2_new)
print(a2_age_predicted)
# Age cluster 2, 1


# In[242]:


# perform predictions on the model:(income 100,Spending score 80),(income 75,Spending score 20),(income 135,Spending score 15)
i1_new = [[100, 80], [75, 20], [135, 15]]
i1_income_predicted = km_income.predict(i1_new)
print(i1_income_predicted)
# Age cluster 0, 2, 2


# In[243]:


# perform predictions on the model:(income 25,Spending score 42),(income 25,Spending score 50),(income 15,Spending score 10)
i2_new = [[35, 42], [25, 50], [15, 10]]
i2_income_predicted = km_income.predict(i2_new)
print(i2_income_predicted)
# Age cluster 1, 4, 4


# In[ ]:




