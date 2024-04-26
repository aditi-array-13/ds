#!/usr/bin/env python
# coding: utf-8

# # Data Analytics III

# ### 1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset.
# ### 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall on the given dataset.

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import mstats
import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


cols = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
file_path = r'C:\Users\Aditi\Documents\sem 6 assignments\Data science assignment\Iris.csv'
df = pd.read_csv(file_path, sep=',', names=cols)

print("Rows and Column of Data : ",df.shape)
df


# ### Check for missing values

# In[23]:


missing_values = df.isnull().sum()
missing_values


# ### Get initial statistics

# In[24]:


data_description = df.describe()
print("Data Description : ")
data_description


# ### Variable descriptions

# In[25]:


variable_descriptions = df.dtypes
print('Variable Descriptions : ')
variable_descriptions


# ### Check dimensions

# In[26]:


data_dimensions = df.shape
print('Data Dimensions : ')
data_dimensions


# ### Handle outliers

# In[27]:


# Apply winsorization to cap extreme values at the 1st and 99th percentiles
df_winsorized = df.copy()
for col in df_winsorized.columns[1:5]:  # Exclude 'Id' and 'Species'
    df_winsorized[col] = mstats.winsorize(df_winsorized[col], limits=[0.01, 0.01])


# ### Split the data into train and test sets

# In[28]:


X = df_winsorized.drop(['Id', 'Species'], axis=1)  # Features
y = df_winsorized['Species']  # Target variable


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Build the Naive Bayes classifier

# In[30]:


model = GaussianNB()
model.fit(X_train, y_train)


# ### Make predictions

# In[31]:


y_pred = model.predict(X_test)


# ### Evaluate the model

# In[32]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)


# In[33]:


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)


# In[34]:


categories = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# In[35]:


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




