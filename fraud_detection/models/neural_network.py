# neural_network.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_neural_network(input_shape):
    """
    Create a neural network model for fraud detection.

    Parameters:
    - input_shape: Tuple representing the shape of the input data.

    Returns:
    - model: Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))  # Increased layer size for better learning
    model.add(Dropout(0.3))  # Increased dropout rate to prevent overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
