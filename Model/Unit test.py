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
from Validation import accuracy

# Test Class
class TestDiabetesModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests."""
        cls.predictions = model.predict(X_test_std)

    def test_output_range(self):
        """Test if model output is between [0, 1] for test samples."""
        self.assertTrue(np.all((self.predictions >= 0) & (self.predictions <= 1)), 
                        f"Some outputs are out of model prediction range [0,1]: {self.predictions}")

    def test_prediction_accuracy(self):
        """Test if model predicts correctly based on the test set."""
        self.assertGreaterEqual(accuracy, 0.6, "Model accuracy is below 60%.")

if __name__ == '__main__':
    unittest.main()
