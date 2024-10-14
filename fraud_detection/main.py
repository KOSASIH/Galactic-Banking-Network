# main.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from utils.data_preprocessing import preprocess_data
from utils.feature_extraction import extract_features
from models.neural_network import create_neural_network
from models.training import train_model
from config import *

def main():
    # Load data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    # Preprocess data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Extract features
    train_features = extract_features(train_data, method='pca', n_components=2)
    test_features = extract_features(test_data, method='pca', n_components=2)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_data['target'], test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    # Create and train the neural network model
    model = create_neural_network(input_shape=(2,))
    model = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model on the test set
    y_pred = model.predict(test_features)
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(test_data['target'], y_pred_class)
    print('Test Accuracy:', accuracy)
    print('Test Classification Report:')
    print(classification_report(test_data['target'], y_pred_class))

    # Save the trained model
    model.save(MODEL_PATH)

if __name__ == '__main__':
    main()
