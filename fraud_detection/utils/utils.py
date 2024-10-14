# utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
    - data: Data to be split.
    - test_size: Proportion of data to be used for testing (default: 0.2).
    - random_state: Random seed for reproducibility (default: 42).

    Returns:
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training labels.
    - y_test: Testing labels.
    """
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def save_data(data, data_path):
    """
    Save data to a CSV file.

    Parameters:
    - data: Data to be saved.
    - data_path: Path to the CSV file.
    """
    data.to_csv(data_path, index=False)
