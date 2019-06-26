#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import itertools


# In[2]:


train = pd.read_csv("D://projects/titanic/train.csv")
test  = pd.read_csv("D://projects/titanic/test.csv")


# In[3]:


features = ['Sex', 'Pclass', 'Fare', 'Age', 'Embarked', 'SibSp', 'Parch']
data = train[features]
x = test[features]
target = train['Survived']


# In[4]:


data.info()


# In[5]:


x.info()


# In[6]:


data['Embarked'].fillna('S', inplace=True)
x['Embarked'].fillna('S', inplace=True)
data['Age'].fillna(data['Age'].mean(), inplace=True)
x['Age'].fillna(x['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
x['Fare'].fillna(x['Fare'].mean(), inplace=True)


# In[7]:


data.info()


# In[8]:


x.info()


# In[9]:


from sklearn.feature_extraction import DictVectorizer


# In[10]:


dict_vec = DictVectorizer(sparse = False)
data = dict_vec.fit_transform(data.to_dict(orient = 'record'))
dict_vec.feature_names_


# In[11]:


x = dict_vec.fit_transform(x.to_dict(orient = 'record'))


# In[12]:


from sklearn import svm


# In[13]:


model = svm.SVC()


# In[14]:


model.fit(data,target)


# In[21]:


y=model.predict(x)
y


# In[22]:


submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y})
submission.to_csv('submission.csv', index = False)


# In[23]:


df= pd.read_csv('submission.csv')
df.head()


# In[24]:


df.mean()

