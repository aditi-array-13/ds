#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling I

# ## Perform the following operations using Python on any open source dataset (e.g., data.csv)
# 1. Import all the required Python Libraries.
# 2. Locate open source data from the web (e.g., https://www.kaggle.com). Provide a clear 
#    description of the data and its source (i.e., URL of the web site).
# 3. Load the Dataset into pandas dataframe.
# 4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe() 
#    function to get some initial statistics. Provide variable descriptions. Types of variables etc. 
#    Check the dimensions of the data frame.
# 5. Data Formatting and Data Normalization: Summarize the types of variables by checking the 
#    data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set. 
#    If variables are not in the correct data type, apply proper type conversions.
# 6. Turn categorical variables into quantitative variables in Python.
#    In addition to the codes and outputs, explain every operation that you do in the above steps and explain 
#    everything that you do to import/read/scrape the data set.

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("C:\\Users\\Aditi\\Documents\\sem 6 assignments\\Data science assignment\\most-popular.csv")


# In[4]:


# Display the initial dataset
print("Initial Dataset:")
print(df)
print("\n")


# In[5]:


# Check for missing values
missing_values = df.isnull().sum()
missing_values


# In[7]:


# Get initial statistics
data_description = df.describe()
data_description


# In[9]:


# Variable descriptions
variable_descriptions = df.dtypes
variable_descriptions


# In[10]:


# Check dimensions
data_dimensions = df.shape
data_dimensions


# In[12]:


# If needed, convert variables to the correct data types
df['season_title'] = pd.to_numeric(df['season_title'], errors='coerce')
# Display variable types
print(variable_descriptions)


# In[13]:


import pandas as pd
# Assuming ' rank' is a categorical variable
df = pd.get_dummies(df, columns=['rank']) # convert categorical values into indicator variable(0, 1)


# In[14]:


print(df)

