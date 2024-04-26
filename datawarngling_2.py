#!/usr/bin/env python
# coding: utf-8

# #  Data Wrangling II

# ## Create an “Academic performance” dataset of students and perform the following operations using Python.
# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or 
#    inconsistencies, use any of the suitable techniques to deal with them.
# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques 
#    to deal with them.
# 3. Apply data transformations on at least one of the variables. The purpose of this 
#    transformation should be one of the following reasons: to change the scale for better 
#    understanding of the variable, to convert a non-linear relation into a linear one, or to decrease 
#    the skewness and convert the distribution into a normal distribution.
#    
#    Reason and document your approach properly.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


Student_data = {
    'Student_ID': range(1, 51),
    'Name': [f'Student_{i}' for i in range(1, 51)],
    'Age': np.random.randint(18, 25, size=50),
    'Gender': np.random.choice(['Male', 'Female'], size=50),
    'Math_Score': np.random.randint(0, 100, size=50),
    'Physics_score': np.random.randint(0, 100, size=50),
    'ComputerScience_score': np.random.randint(0, 100, size=50),
    'Absenteeism': np.random.choice([0, 1], size=50)
}


# In[46]:


df = pd.DataFrame(Student_data)


# In[48]:


# Introduce missing values
df.loc[df['Absenteeism'] == 0, 'Math_Score'] = np.nan


# In[49]:


# Display the initial dataset
print("Initial Dataset:")
df


# In[50]:


# Information about the dataset
df.info()


# In[51]:


#checking the missing values
df.isnull().sum()


# In[52]:


# Data cleaning
# Handling missing values
df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)


# In[53]:


# Display the dataset after handling missing values
print("\nDataset after Handling Missing Values:")
df


# # Outliers

# In[54]:


sns.boxplot(y=df['Math_Score'])


# In[55]:


sns.boxplot(y=df['Physics_score'])


# In[56]:


sns.boxplot(y=df['ComputerScience_score'])


# In[57]:


# Visualize outliers using boxplots
sns.boxplot(data=df[['Math_Score', 'Physics_score', 'ComputerScience_score']])
plt.title("Boxplots of Scores")
plt.show()


# In[58]:


# Identify and handle outliers using Z-score
z_scores = np.abs((df[['Math_Score', 'Physics_score', 'ComputerScience_score']] - df[['Math_Score', 'Physics_score', 'ComputerScience_score']].mean()) / df[['Math_Score', 'Physics_score', 'ComputerScience_score']].std())
outliers = z_scores > 2


# In[59]:


# Remove outliers
df = df[~outliers.any(axis=1)]


# In[60]:


# Display the dataset after handling outliers
print("\nDataset after Handling Numeric Outliers:")
df


# In[61]:


# Data Transformation
# Transform the data according to the given question
df['Final_Result'] = np.where(
    (df['Math_Score'] >= 20) & (df['Physics_score'] >= 15) & (df['ComputerScience_score'] >= 15) & (df['Absenteeism'] == 1),
    'Pass', 'Fail'
)


# In[62]:


# Display the dataset after data transformation and adding 'Final_Result'
print("\nDataset after Data Transformation and Adding 'Final_Result':")
df


# In[63]:


#Another Data transformation
#Apply a square root transformation on Physics_score
df['Transformed_Physics_Score'] = np.sqrt(df['Physics_score'])


# In[20]:


df


# In[64]:


from scipy.stats import skew

# Calculate skewness before transformation
original_skewness = skew(df["ComputerScience_score"])

# Apply transformation (e.g., Square Root Transformation)
transformed_data = np.sqrt(df["ComputerScience_score"])

# Calculate skewness after transformation
transformed_skewness = skew(transformed_data)

print("Original Skewness:", original_skewness)
print("Transformed Skewness:", transformed_skewness)


# To decrease skewness and transform a distribution closer to a normal distribution, you can use techniques like the Box-Cox transformation or Yeo-Johnson transformation. These transformations are specifically designed to stabilize variance and make the data more normal. Here's an example using the Box-Cox transformation:

# In[65]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew, norm
import seaborn as sns


# In[66]:


# Generating a skewed dataset
np.random.seed(42)
skewed_data = np.random.exponential(size=100)

skewed_data


# In[67]:


# Calculate skewness before transformation
skewness_before = skew(skewed_data)

skewness_before


# In[68]:


# Apply Box-Cox transformation
transformed_data, lambda_value = boxcox(skewed_data + 1)  # Adding 1 to handle zero values if present


# In[69]:


# Calculate skewness after transformation
skewness_after = skew(transformed_data)

skewness_after


# In[70]:


# Plotting original and transformed distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(skewed_data, kde=True)
plt.title(f'Skewed Distribution (Skewness: {skewness_before:.2f})')


# In[71]:


plt.subplot(1, 2, 2)
sns.histplot(transformed_data, kde=True)
plt.title(f'Transformed Distribution (Skewness: {skewness_after:.2f})')


# In[75]:


plt.tight_layout()
plt.show()


# In[74]:


df


# In[ ]:




