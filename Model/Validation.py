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
from NN import *

early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta = 0.001,
    patience=10,    
    restore_best_weights=True 
)

history = model.fit(X_train_std, y_train, epochs = 100, validation_data = (X_valid_std, y_valid),
                    batch_size=1, verbose = 0, callbacks = [early_stopping])


# In[5]:


loss, accuracy = model.evaluate(X_test_std, y_test)
print(f"Model accuracy: {accuracy}")
print(f"Model loss: {loss}")


# In[6]:


y_pred_prob = model.predict(X_test_std)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)


# In[7]:


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[8]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()


# # Roc Curve

# In[10]:


#plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label='AUC = %0.4f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# # Confusion Matrix

# In[12]:


cm = confusion_matrix(y_test, y_pred)

labels = {0:'No diabetes',
         1:'Diabetes'}

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = [labels[i] for i in range(len(labels))])

disp.plot(colorbar = False)

plt.title('Confusion Matrix for Diabetes predictions')
plt.show()


# In[13]:


# Compute confusion matrix
tn, fp, fn, tp = cm.ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f'Sensitivity: {sensitivity:.3f}') # Ability to detect diabetes
print(f'Specificity: {specificity:.3f}') # False positives

