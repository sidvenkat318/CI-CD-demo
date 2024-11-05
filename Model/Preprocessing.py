#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


# In[2]:


# Loading dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, header=None)
columns = ['Pregnancies', 'Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','Pedigree','Age','Diabetes']
df.columns = columns
df


# In[3]:


df1 = df.iloc[:,:-1]
zeros = df1[(df1 == 0).any(axis=1)]
zeros


# In[4]:


df2 = df1.drop(columns = ['Insulin'])
df2


# In[5]:


filtered_df = df2[df2['Glucose'] != 0]
med_glucose = filtered_df['Glucose'].median()
med_glucose


# In[6]:


df3 = df2.copy()
df3['Glucose'] = df3['Glucose'].replace(0, med_glucose)


# In[7]:


filtered_df = df2[df2['Blood Pressure'] != 0]
med_bp = filtered_df['Blood Pressure'].median()
med_bp


# In[8]:


df3['Blood Pressure'] = df3['Blood Pressure'].replace(0, med_bp)


# In[9]:


filtered_df = df2[df2['BMI'] != 0]
mean_bmi = filtered_df['BMI'].mean()
mean_bmi


# In[10]:


df3['BMI'] = df3['BMI'].replace(0, mean_bmi)


# In[11]:


median_skin = filtered_df['Skin Thickness'].median()
median_skin


# In[12]:


df3['Skin Thickness'] = df3['Skin Thickness'].replace(0, median_skin)


# In[13]:


med_ped = np.median(filtered_df['Pedigree'])
med_ped


# In[14]:


df3['Pedigree'] = df3['Pedigree'].replace(0, med_ped)


# In[15]:


filtered_df = df2[df2['Age'] != 0]
med_age = filtered_df['Age'].median()
med_age


# In[16]:


df3['Age'] = df3['Age'].replace(0, med_age)


# In[17]:


df4 = pd.concat([df3, df.iloc[:,-1]], axis=1)
df4

