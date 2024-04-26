#!/usr/bin/env python
# coding: utf-8

# # Descriptive Statistics - Measures of Central Tendency and variability

# ## Perform the following operations on any open source dataset (e.g., data.csv)
# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a 
#    dataset (age, income etc.) with numeric variables grouped by one of the qualitative 
#    (categorical) variable. For example, if your categorical variable is age groups and quantitative 
#    variable is income, then provide summary statistics of income grouped by the age groups. 
#    Create a list that contains a numeric value for each response to the categorical variable.
# 2. Write a Python program to display some basic statistical details like percentile, mean, 
#    standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of 
#    iris.csv dataset.

# In[1]:


import numpy as np 
import pandas as pd
from statistics import mean, median, mode


# # Importing Data and reading into a Pandas DataFrame

# In[2]:


cols = ['ID','Age','Experience','Income','Family','Education','Personal Loan','CreditCard']
file_path = r'C:\Users\Aditi\Documents\sem 6 assignments\Data science assignment\Bank_Personal_Loan_Modelling - Bank_Personal_Loan_Modelling.csv'
df = pd.read_csv(file_path, names=cols)

print("Rows and Column of Data : ",df.shape)
df


# In[3]:


# Group by 'Education' and calculate summary statistics for 'Age' and 'Income'
summary_statistics = df.groupby('Education')[['Age', 'Income']].describe()


# In[4]:


# Display the result in a formatted way
print("Summary Statistics for 'Age' and 'Income' Grouped by 'Education':")
summary_statistics


# - Converted categorical Variable into numerical ones by mapping 

# In[5]:


# Create a list that contains a numeric value for each response to the categorical variable 'Education'
education_mapping = {'1': 1, '2': 2, '3': 3}
df['Education_Numeric'] = df['Education'].map(education_mapping)


# In[6]:


# Display the dataset with the numeric values for the categorical variable
df[['Education', 'Education_Numeric']]


# #  2. Write a Python program to display some basic statistical details like percentile, mean, standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of iris.csv dataset.

# In[7]:


cols = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
file_path = r'C:\Users\Aditi\Documents\sem 6 assignments\Data science assignment\Iris.csv'
df = pd.read_csv(file_path, names=cols)

print("Rows and Column of Data : ",df.shape)
df


# In[11]:


# Display basic statistical details for 'Iris-setosa'
setosa_stats = df[df['Species'] == 'Iris-setosa']
print("Statistical details for Iris-setosa:")
setosa_stats.describe()


# In[12]:


# Display basic statistical details for 'Iris-versicolor'
versicolor_stats = df[df['Species'] == 'Iris-versicolor']
print("\nStatistical details for Iris-versicolor:")
versicolor_stats.describe()


# In[13]:


# Display basic statistical details for 'Iris-virginica'
virginica_stats = df[df['Species'] == 'Iris-virginica']
print("\nStatistical details for Iris-virginica:")
virginica_stats.describe()


# In[ ]:




