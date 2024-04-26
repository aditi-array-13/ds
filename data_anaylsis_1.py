#!/usr/bin/env python
# coding: utf-8

# # Data Analytics I
# 

# ## Create a Linear Regression Model using Python/R to predict home prices using Boston Housing  Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains information about various houses in Boston through different parameters. There are 506 samples and 14 feature variables in this dataset.

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[2]:


cols = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv']
file_path = r'C:\Users\Aditi\Documents\sem 6 assignments\Data science assignment\BostonHousing.csv'
df = pd.read_csv(file_path, sep=',', names=cols)

print("Rows and Column of Data : ",df.shape)
df


# ### Check for missing values

# In[3]:


missing_values = df.isnull().sum()
missing_values


# ### Get initial statistics

# In[4]:


data_description = df.describe()
print("Data Description : ")
data_description


# ### Variable descriptions

# In[5]:


variable_descriptions = df.dtypes
print('Variable Descriptions : ')
variable_descriptions


# ### Check dimensions

# In[6]:


data_dimensions = df.shape
print('Data Dimensions : ')
data_dimensions


# ## Data Preprocessing

# In[7]:


# Drop rows with missing values
df.dropna(inplace=True)


# In[8]:


# Convert 'tax' column to numeric, coercing errors to NaN
df['tax'] = pd.to_numeric(df['tax'], errors='coerce')


# In[9]:


# Check for any remaining non-numeric values
non_numeric_values = df['tax'].loc[df['tax'].apply(lambda x: isinstance(x, str))]
print(non_numeric_values)


# In[10]:


# Handle missing values or non-numeric values as needed
# To remove rows with missing or non-numeric values:
df.dropna(subset=['tax'], inplace=True)


# In[11]:


# Now proceed with imputing missing values or other data cleaning steps
df['tax'].fillna(df['tax'].mean(), inplace=True)


# In[12]:


df


# ### Handle Outliers

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()

# Handle outliers using winsorization
from scipy.stats.mstats import winsorize

# Winsorize 'medv' column to limit outliers
df['medv'] = winsorize(df['medv'], limits=[0.05, 0.05])


# ### Data Transformation

# In[15]:


# Convert categorical variables to numeric using one-hot encoding (if applicable)
df = pd.get_dummies(df, columns=['chas'])

# Perform scaling or normalization (example using Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']] = scaler.fit_transform(df[['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']])


# In[16]:


df


# ### Split the Data

# In[17]:


X = df.drop('medv', axis=1)  # features
y = df['medv']  # target variable


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Train the Linear Regression Model

# In[19]:


model = LinearRegression()
model.fit(X_train, y_train)


# ### Evaluate the Model

# In[20]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# ### Make Predictions

# In[21]:


# Make predictions on new data
new_data = pd.DataFrame({'crim': [0.1], 'zn': [20], 'indus': [6], 'nox': [0.5], 'rm': [6], 'age': [60], 
                         'dis': [6], 'rad': [4], 'tax': [300], 'ptratio': [15], 'b': [350], 'lstat': [10],
                         'chas_0': [1], 'chas_1': [0]})


# In[22]:


predicted_price = model.predict(new_data)
print('Predicted price for new data:', predicted_price)


# In[ ]:




