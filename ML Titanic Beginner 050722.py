#thank you to https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy. 
#I followed their code and completed the code on my jupiter notebook to get my results
#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# To predict survival of passengers from the Titanic Dataset

# In[273]:


import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import random
import time
import warnings
from subprocess import check_output
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


# In[274]:


import matplotlib.pylab as pylab
import seaborn as sns


# In[275]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize']=12,8


# In[276]:


data_raw=pd.read_csv('Documents/tt_train.csv')


# In[277]:


data_val=pd.read_csv('Documents/tt_test.csv')


# In[278]:


data1=data_raw.copy(deep=True)


# In[279]:


data_cleaner=[data1, data_val]


# In[280]:


print(data_raw.info())


# In[281]:


print(data_raw.sample(10))


# In[282]:


data_raw.sample(10)


# In[283]:


data_raw.describe()


# In[284]:


print('Train columns with null values:\n',data1.isnull().sum())
print("-"*10)


# In[285]:


print('Test/Validation columns with null values:\n', data_val.isnull().sum()
)
print("-"*10)


# In[286]:


data_raw.describe()


# In[287]:


data_raw.describe(include='all')


# In[288]:


for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)


# In[289]:


dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)


# In[290]:


dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)


# In[291]:


drop_column=['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace=True)


# In[325]:


drop_column_val=['PassengerId', 'Cabin', 'Ticket']
data_val.drop(drop_column_val, axis=1, inplace=True)


# In[326]:


print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())


# In[327]:


dataset['Embarked'].fillna(data1['Embarked'].mode()[0], inplace=True)


# In[328]:


print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())


# In[295]:


data1.head()


# In[329]:


dataset['Embarked'].fillna(data_val['Embarked'].mode()[0], inplace=True)


# In[330]:


for dataset in data_cleaner:
#    dataset['FamilySize']= dataset['SibSp'] + dataset['Parch']+1
#    dataset['IsAlone'] = 1
#    dataset['IsAlone'].loc[dataset['FamilySize']>1]=0
    
#    dataset['Title']= dataset['Name'].str.split(", ", expand=True[1].str.split(", ",expand=True)[0])
    dataset['FareBin']= pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin']= pd.cut(dataset['Age'].astype(int), 5)


# In[331]:


data1.describe(include= 'all')


# In[332]:


data_val.head()


# In[333]:


data1['FamilySize']= data1['SibSp']+data1['Parch']+1
print(data1['FamilySize'])


# In[334]:


data_val['FamilySize']=data_val['SibSp']+data_val['Parch']+1
print(data_val['FamilySize'])


# In[335]:


data1['IsAlone']= 1
data_val['IsAlone']=1
#data1['IsAlone'].loc[data1['FamilySize']>1]=0
#data_val['IsAlone'].loc[data_val['FamilySize']>1]=0


# In[336]:


data1.head()


# In[337]:


data1.loc[data1["FamilySize"]>1, "IsAlone"]=1
data1.loc[data1["FamilySize"]<=1, "IsAlone"]=0


# In[338]:


data_val.loc[data_val["FamilySize"]>1, "IsAlone"]=1
data_val.loc[data_val["FamilySize"]<=1, "IsAlone"]=0


# In[339]:


#dataset['Title']= dataset['Name'].str.split(", ", expand=True[1].str.split(", ",expand=True)[0])


# In[340]:


data1['Title']=data1["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[341]:


data1.Title.head(10)


# In[342]:


data_val['Title']=data_val["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[343]:


data_val.Title.head(10)


# In[344]:


print(data1['Title'].value_counts())


# In[345]:


stat_min=10


# In[346]:


title_names=(data1['Title'].value_counts()<stat_min)


# In[347]:


data1['Title']=data1['Title'].replace("title_names","Misc")


# In[348]:


print(data1['Title'].value_counts())


# In[349]:


#new = s.apply(lambda num: num+5)
data1['Title']= data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x)
print(data1['Title'].value_counts())


# In[350]:


data1.info()
data_val.info()
data1.sample(5)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

label = preprocessing.LabelEncoder()
# In[317]:


data1.head()


# In[351]:


for dataset in data_cleaner:
    dataset['Sex_Code']= label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code']=label.fit_transform(dataset['Embarked'])
    dataset['Title_Code']=label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code']=label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code']=label.fit_transform(dataset['FareBin'])
#data_val['Sex_Code']= labelencoder.fit(data_val['Sex'])
#data_val['Embarked_Code']=labelencoder.fit_transform(data_val['Embarked'])
#data_val['Title_Code']=labelencoder.fit_transform(data_val['Title'])
#data_val['AgeBin_Code']=labelencoder.fit_transform(data_val['AgeBin'])
#data_val['FareBin_Code']=labelencoder.fit_transform(data_val['FareBin'])


# In[352]:


data1.dtypes


# In[353]:


Target = ['Survived']


# In[ ]:





# In[354]:


data1['Age']=data1['Age'].astype('int64')


# In[355]:


data1.info()


# In[356]:


data_val['Age']=data_val['Age'].astype('int64')


# In[357]:


data1.head()


# In[358]:


data1.describe(include='all')


# In[359]:


data1['Sex_female'] = 1
data1.loc[data1["Sex"]=="female", "Sex_female"]=1
data1.loc[data1["Sex"]=="male", "Sex_female"]=0


# In[360]:


data1.head()


# In[361]:


data1["Sex_male"]=1
data1.loc[data1["Sex"]=="male", "Sex_male"]=1
data1.loc[data1["Sex"]=="female", "Sex_male"]=0


# In[362]:


data_val["Sex_female"]=1
data_val.loc[data_val["Sex"]=="female", "Sex_female"]=1
data_val.loc[data_val["Sex"]=="male", "Sex_female"]=0
data_val["Sex_male"]=1
data_val.loc[data_val["Sex"]=="male", "Sex_male"]=1
data_val.loc[data_val["Sex"]=="female", "Sex_male"]=0


# In[363]:


data1['Embarked'].describe()


# In[364]:


print(data1['Embarked'].value_counts())


# In[365]:


data1["Embarked_S"]=1
data1.loc[data1["Embarked"]=="S", "Embarked_S"]=1
data1.loc[data1["Embarked"]=="C", "Embarked_S"]=0
data1.loc[data1["Embarked"]=="Q", "Embarked_S"]=0
data1.loc[data1["Embarked"]=="NaN", "Embarked_S"]=1
data1["Embarked_C"]=1
data1.loc[data1["Embarked"]=="C", "Embarked_C"]=1
data1.loc[data1["Embarked"]=="Q", "Embarked_C"]=0
data1.loc[data1["Embarked"]=="S", "Embarked_C"]=0
data1.loc[data1["Embarked"]=="NaN", "Embarked_C"]=0
data1["Embarked_Q"]=1
data1.loc[data1["Embarked"]=="Q", "Embarked_Q"]=1
data1.loc[data1["Embarked"]=="S", "Embarked_Q"]=0
data1.loc[data1["Embarked"]=="C", "Embarked_Q"]=0
data1.loc[data1["Embarked"]=="Nan", "Embarked_Q"]=0


# In[366]:


data1.head()


# In[367]:


data_val["Embarked_S"]=1
data_val.loc[data_val["Embarked"]=="S", "Embarked_S"]=1
data_val.loc[data_val["Embarked"]=="C", "Embarked_S"]=0
data_val.loc[data_val["Embarked"]=="Q", "Embarked_S"]=0
data_val.loc[data_val["Embarked"]=="NaN", "Embarked_S"]=1
data_val["Embarked_C"]=1
data_val.loc[data_val["Embarked"]=="C", "Embarked_C"]=1
data_val.loc[data_val["Embarked"]=="Q", "Embarked_C"]=0
data_val.loc[data_val["Embarked"]=="S", "Embarked_C"]=0
data_val.loc[data_val["Embarked"]=="NaN", "Embarked_C"]=0
data_val["Embarked_Q"]=1
data_val.loc[data_val["Embarked"]=="Q", "Embarked_Q"]=1
data_val.loc[data_val["Embarked"]=="S", "Embarked_Q"]=0
data_val.loc[data_val["Embarked"]=="C", "Embarked_Q"]=0
data_val.loc[data_val["Embarked"]=="Nan", "Embarked_Q"]=0


# In[368]:


data1["Title_Mr"]=1
data1.loc[data1["Title"]=="Mr", "Title_Mr"]=1
data1.loc[data1["Title"]=="Miss", "Title_Mr"]=0
data1.loc[data1["Title"]=="Mrs", "Title_Mr"]=0
data1.loc[data1["Title"]=="Master", "Title_Mr"]=0
data1.loc[data1["Title"]=="Misc", "Title_Mr"]=0
data1["Title_Miss"]=1
data1.loc[data1["Title"]=="Mr", "Title_Miss"]=0
data1.loc[data1["Title"]=="Miss", "Title_Miss"]=1
data1.loc[data1["Title"]=="Mrs", "Title_Miss"]=0
data1.loc[data1["Title"]=="Master", "Title_Miss"]=0
data1.loc[data1["Title"]=="Misc", "Title_Miss"]=0
data1["Title_Mrs"]=1
data1.loc[data1["Title"]=="Mr", "Title_Mrs"]=0
data1.loc[data1["Title"]=="Miss", "Title_Mrs"]=0
data1.loc[data1["Title"]=="Mrs", "Title_Mrs"]=1
data1.loc[data1["Title"]=="Master", "Title_Mrs"]=0
data1.loc[data1["Title"]=="Misc", "Title_Mrs"]=0
data1["Title_Master"]=1
data1.loc[data1["Title"]=="Mr", "Title_Master"]=0
data1.loc[data1["Title"]=="Miss", "Title_Master"]=0
data1.loc[data1["Title"]=="Mrs", "Title_Master"]=0
data1.loc[data1["Title"]=="Master", "Title_Master"]=1
data1.loc[data1["Title"]=="Misc", "Title_Master"]=0
data1["Title_Misc"]=1
data1.loc[data1["Title"]=="Mr", "Title_Misc"]=0
data1.loc[data1["Title"]=="Miss", "Title_Misc"]=0
data1.loc[data1["Title"]=="Mrs", "Title_Misc"]=0
data1.loc[data1["Title"]=="Master", "Title_Misc"]=0
data1.loc[data1["Title"]=="Misc", "Title_Misc"]=1


# In[369]:


data1.head()


# In[370]:


data_val["Title_Mr"]=1
data_val.loc[data_val["Title"]=="Mr", "Title_Mr"]=1
data_val.loc[data_val["Title"]=="Miss", "Title_Mr"]=0
data_val.loc[data_val["Title"]=="Mrs", "Title_Mr"]=0
data_val.loc[data_val["Title"]=="Master", "Title_Mr"]=0
data_val.loc[data_val["Title"]=="Misc", "Title_Mr"]=0
data_val["Title_Miss"]=1
data_val.loc[data_val["Title"]=="Mr", "Title_Miss"]=0
data_val.loc[data_val["Title"]=="Miss", "Title_Miss"]=1
data_val.loc[data_val["Title"]=="Mrs", "Title_Miss"]=0
data_val.loc[data_val["Title"]=="Master", "Title_Miss"]=0
data_val.loc[data_val["Title"]=="Misc", "Title_Miss"]=0
data_val["Title_Mrs"]=1
data_val.loc[data_val["Title"]=="Mr", "Title_Mrs"]=0
data_val.loc[data_val["Title"]=="Miss", "Title_Mrs"]=0
data_val.loc[data_val["Title"]=="Mrs", "Title_Mrs"]=1
data_val.loc[data_val["Title"]=="Master", "Title_Mrs"]=0
data_val.loc[data_val["Title"]=="Misc", "Title_Mrs"]=0
data_val["Title_Master"]=1
data_val.loc[data_val["Title"]=="Mr", "Title_Master"]=0
data_val.loc[data_val["Title"]=="Miss", "Title_Master"]=0
data_val.loc[data_val["Title"]=="Mrs", "Title_Master"]=0
data_val.loc[data_val["Title"]=="Master", "Title_Master"]=1
data_val.loc[data_val["Title"]=="Misc", "Title_Master"]=0
data_val["Title_Misc"]=1
data_val.loc[data_val["Title"]=="Mr", "Title_Misc"]=0
data_val.loc[data_val["Title"]=="Miss", "Title_Misc"]=0
data_val.loc[data_val["Title"]=="Mrs", "Title_Misc"]=0
data_val.loc[data_val["Title"]=="Master", "Title_Misc"]=0
data_val.loc[data_val["Title"]=="Misc", "Title_Misc"]=1


# In[371]:


print("Train columns with null values: \n", data_val.isnull().sum())
print("-"*10)
print(data1.info())
print("-"*10)


# In[372]:


data1[data1.isna().any(axis=1)]


# In[373]:


data_val[data_val.isna().any(axis=1)]


# In[374]:


print('Train columns with null values: \n', data1.isnull().sum())
print("-"*10)
print(data1.info())
print("-"*10)


# In[376]:


print("Test/Validation columns with null values:\n", data_val.isnull().sum())
print("-"*10)
print(data_val.info())
print("-")


# In[377]:


data_raw.describe(include='all')


# unsure how to proceed since the data was split already. I guess I fit the data1? and predict the data_val?  I will try that

# In[379]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[382]:


X=data1.drop('Survived', axis=1).values
y=data1['Survived'].values


# In[383]:


reg=LinearRegression()
reg.fit(X,y)


# In[386]:


drop_column=['Name']
data1.drop(drop_column, axis=1, inplace=True)
drop_column_val=['Name']
data_val.drop(drop_column_val,axis=1, inplace=True)


# In[388]:


data1.head()


# In[389]:


drop_column=['Sex']
data1.drop(drop_column, axis=1, inplace=True)
drop_column_val=['Sex']
data_val.drop(drop_column_val,axis=1, inplace=True)


# In[390]:


data1.head()


# In[393]:


data1.info()


# In[399]:


X_age=data1['Age'].values
y_age=data1['Survived'].values


# In[400]:


print("Dimensions of y before reshaping: ", y_age.shape)
print("Dimensions of X before reshaping: ", X_age.shape)


# In[401]:


y_reshaped_age=y.reshape(-1,1)
X_reshaped_age=X.reshape(-1,1)


# In[404]:


print("Dimensions of y   reshaping: ", y_reshaped_age.shape)
print("Dimensions of X   reshaping: ", X_reshaped_age.shape)


# In[405]:


reg_age=LinearRegression()
reg_age.fit(X_reshaped_age, y_reshaped_age)
#y_prediction=reg_age.predict(X_age)


# In[406]:


plt.subplot()
plt.boxplot(data1['Age'], showmeans=True, meanline=True )
plt.title("Age")
plt.show()


# In[409]:


sns.barplot(x='Embarked', y='Survived', data = data1)
plt.show()


# In[411]:


sns.barplot(x='IsAlone', y='Survived', data = data1)
plt.show()


# In[413]:


sns.pointplot(x='FareBin', y='Survived', data = data1)
plt.show()


# In[416]:


pp = sns.pairplot(data1, hue='Survived', palette='deep', height=1.2, diag_kind='kde')
pp.set(xticklabels=[])
plt.show()


# In[422]:


sns.heatmap(data1.corr(), square=True, cbar_kws={'shrink':.9}, annot=True, linewidths=0.1, vmax=1.0, linecolor='white', annot_kws={'fontsize':12})
plt.title("Pearson Correlation of Features", y=1.05, size=15)
plt.show()


# In[ ]:





# In[ ]:




