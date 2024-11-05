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


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


# In[15]:


def run_notebook_and_get_df(notebook_path, target_variable):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)

    global_vars = {'pd': pd,
                  'np':np,
                  'sns':sns,
                  'tf':tf,
                  'plt':plt}
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(notebook, {'metadata': {'path': './'}})
    
    # Execute each cell in the notebook
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(cell.source, global_vars)
    
    # Return the specific DataFrame
    return global_vars.get(target_variable)


# Run the notebook and get the df4 DataFrame
notebook_path = 'C:/Users/Sid/Desktop/DSOR 752/CI-CD demo/CI-CD-demo/Model/Preprocessing/Preprocessing.ipynb'
df4 = run_notebook_and_get_df(notebook_path, 'df4')
print(df4)


# ## Splitting data

# In[20]:


x = df4.iloc[:,:-1]
y = df4.iloc[:,-1]

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=43)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)
print(y_train.value_counts())


# In[21]:


from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=43)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
print(y_train.value_counts())


# In[22]:


scaler = StandardScaler()
scaler.fit(X_train)


# In[23]:


X_train_std = scaler.transform(X_train) 
X_test_std = scaler.transform(X_test)  
X_valid_std = scaler.transform(X_valid) 


# # Neural Network

# In[25]:


n = X_train_std.shape[1]

model = Sequential()

model.add(Dense(64, activation='relu', use_bias=False, bias_initializer='ones', input_shape = (n,))) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

opt = tf.keras.optimizers.Adam(learning_rate=5e-6) 
model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])

