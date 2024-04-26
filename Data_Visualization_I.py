#!/usr/bin/env python
# coding: utf-8

# #  Data Visualization I

# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about 
#    the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we 
#    can find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
#    is distributed by plotting a histogram.
# 

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[63]:


data = pd.read_csv(r'C:\Users\Aditi\Documents\sem 6 assignments\Data science assignment\Titanic-Dataset.csv')
data


# In[64]:


data.shape


# In[65]:


data.describe()


# In[66]:


data.describe(include = 'object')


# In[67]:


data.isnull().sum()


# In[68]:


data['Age'] = data['Age'].fillna(np.mean(data['Age']))


# In[69]:


data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])


# In[70]:


data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# In[71]:


data.isnull().sum()


# In[72]:


# Perform one-hot encoding
Survived_encoded = pd.get_dummies(data['Survived'], prefix='Survived')

# Plot one-hot encoded data
sns.countplot(data=Survived_encoded)


# In[73]:


# Perform one-hot encoding
Pclass_encoded = pd.get_dummies(data['Pclass'], prefix='Pclass')

# Plot one-hot encoded data
sns.countplot(data=Pclass_encoded)


# In[74]:


# Perform one-hot encoding
embarked_encoded = pd.get_dummies(data['Embarked'], prefix='Embarked')

# Plot one-hot encoded data
sns.countplot(data=embarked_encoded)


# In[75]:


# Perform one-hot encoding
Sex_encoded = pd.get_dummies(data['Sex'], prefix='Sex')

# Plot one-hot encoded data
sns.countplot(data=Sex_encoded)


# In[76]:


sns.boxplot(data['Age'])


# In[77]:


sns.boxplot(data['Fare'])


# In[78]:


sns.boxplot(data['Pclass'])


# In[79]:


sns.boxplot(data['SibSp'])


# In[80]:


sns.catplot(x= 'Pclass', y = 'Age', data=data, kind = 'box')


# In[81]:


sns.catplot(x= 'Pclass', y = 'Fare', data=data, kind = 'strip')


# In[82]:


sns.catplot(x= 'Sex', y = 'Fare', data=data, kind = 'strip')


# In[83]:


sns.catplot(x= 'Sex', y = 'Age', data=data, kind = 'strip')


# In[84]:


sns.pairplot(data)


# In[85]:


sns.scatterplot(x = 'Fare', y = 'Pclass', hue = 'Survived', data = data)


# In[86]:


sns.scatterplot(x = 'Survived', y = 'Fare', data = data)


# In[87]:


sns.distplot(data['Age'])


# In[88]:


sns.distplot(data['Fare'])


# In[89]:


sns.jointplot(x = "Survived", y = "Fare", kind = "scatter", data = data)


# In[90]:


tc = data.corr()
sns.heatmap(tc, cmap="YlGnBu")
plt.title('Correlation')


# ### Price of Ticket for each passenger is distributed

# In[91]:


sns.catplot(x='Pclass', y='Fare', data=data, kind='bar')


# In[ ]:




