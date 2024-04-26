#!/usr/bin/env python
# coding: utf-8

# # Data Visualization III

# - Download the Iris flower dataset or any other dataset into a DataFrame. (e.g., https://archive.ics.uci.edu/ml/datasets/Iris ).Scan the dataset and give the inference as:
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
# 3. Create a boxplot for each feature in the dataset.
# 4. Compare distributions and identify outliers.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.describe(include = 'object')


# In[6]:


data.isnull().sum()


# In[7]:


print("\n\nThe features in the dataset are as follows : ")
print("1. Sepal length : ", data['sepal_length'].dtype)
print("2. Sepal width : ", data['sepal_width'].dtype)
print("3. Petal length : ", data['petal_length'].dtype)
print("4. Petal width : ", data['petal_width'].dtype)
print("5. Species : ", data['species'].dtype)


# In[8]:


sns.histplot(x = data['sepal_length'], kde=True)


# In[9]:


sns.histplot(x = data['sepal_width'], kde=True)


# In[10]:


sns.histplot(x = data['petal_length'], kde=True)


# In[11]:


sns.histplot(x = data['petal_width'], kde=True)


# In[12]:


sns.boxplot(data['sepal_length'])


# In[13]:


sns.boxplot(data['sepal_width'])


# In[14]:


sns.boxplot(data['petal_length'])


# In[15]:


sns.boxplot(data['petal_width'])


# In[16]:


sns.boxplot(x='sepal_length',y='species',data=data)


# In[17]:


sns.boxplot(x='petal_length',y='species',data=data)


# In[ ]:




