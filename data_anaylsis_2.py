#!/usr/bin/env python
# coding: utf-8

# # Data Analytics II

# ### 1. Implement logistic regression using Python/R to perform classification on Social_Network_Ads.csv dataset.
# ### 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall on the given dataset.

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import mstats
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


file_path = 'C:/Users/Aditi/Documents/sem 6 assignments/Data science assignment/Social_Network_Ads.csv'
df = pd.read_csv(file_path)


# In[3]:


df


# In[4]:


# Checking for missing values
missing_values = df.isnull().sum()
missing_values


# In[5]:


data_description = df.describe()
print("Data Description : ")
data_description


# In[6]:


# Variable description
variable_descriptions = df.dtypes
print('Variable Descriptions : ')
variable_descriptions


# In[7]:


data_dimensions = df.shape
print('Data Dimensions : ')
data_dimensions


# In[8]:


# Explore the data
print("Rows and Columns of Data:", df.shape)
print("First few rows of the DataFrame:")
df.head()


# In[9]:


# Handling outliers
df_winsorized = df.copy()
for col in df_winsorized.columns[1:5]:  # Exclude 'Id' and 'Species'
    df_winsorized[col] = mstats.winsorize(df_winsorized[col], limits=[0.01, 0.01])


# In[10]:


# Prepare the data
X = df[['Age', 'EstimatedSalary']]  # Features
y = df['Purchased']  # Target variable


# In[11]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Step 5: Build the Logistic Regression model : predict the dependent variables
from sklearn.linear_model import  LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[17]:


# Step 6: Make predictions
y_pred = model.predict(X_test)


# In[18]:


# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# In[19]:


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[20]:


# Step 8: Compute other performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# In[21]:


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# In[ ]:




