#!/usr/bin/env python
# coding: utf-8

# #  Data Visualization II

# 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution of 
#    age with respect to each gender along with the information about whether they survived or 
#    not. (Column names : 'sex' and 'age')
# 2. Write observations on the inference from the above statistics.

# In[1]:


import seaborn as sns
titanic = sns.load_dataset("titanic")


# In[2]:


titanic


# In[3]:


titanic.head(10)


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# In[6]:


titanic.loc[:,["survived","alive"]]


# In[7]:


#Now Plot boxplot
sns.boxplot(x="sex",y="age",data=titanic)


# In[8]:


sns.boxplot(x="sex",y="age",data=titanic,hue="survived")


# In[ ]:




