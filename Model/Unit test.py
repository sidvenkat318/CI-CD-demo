# Loading libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import resample
import unittest
from NN import *

# Test Class
class TestDiabetesModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the model and relevant variables
        script_path = 'C:/Users/Sid/Desktop/DSOR 752/CI-CD-demo/Model/NN.py'
        cls.variables = run_script_and_get_vars(script_path, ['model', 'X_test_std', 'y_test'])
        cls.model = cls.variables.get('model')
        cls.X_test_std = cls.variables.get('X_test_std')
        cls.y_test = cls.variables.get('y_test')
        
    def test_output_range(self):
        """Test if model output is between [0, 1] for test samples."""
        predictions = self.model.predict(self.X_test_std)
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)), 
                        f"Some outputs are out of model prediction range [0,1]: {predictions}")

    def test_prediction_accuracy(self):
        """Test if model predicts correctly based on the test set."""
        predictions = self.model.predict(self.X_test_std)
        predicted_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == self.y_test)
        self.assertGreaterEqual(accuracy, 0.6, "Model accuracy is below 60%.")

if __name__ == '__main__':
    unittest.main()