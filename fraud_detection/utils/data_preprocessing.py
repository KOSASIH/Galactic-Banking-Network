# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def load_data(data_path):
    """
    Load data from a CSV file.

    Parameters:
    - data_path: Path to the CSV file.

    Returns:
    - data: Loaded data as a Pandas DataFrame.
    """
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    """
    Preprocess data by handling missing values, encoding categorical variables, and scaling numerical features.

    Parameters:
    - data: Data to be preprocessed.

    Returns:
    - preprocessed_data: Preprocessed data.
    """
    # Define preprocessing steps for numerical and categorical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Fit and transform the data
    preprocessed_data = preprocessor.fit_transform(data)

    return preprocessed_data
