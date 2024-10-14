# training.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(model, data, labels):
    """
    Train the neural network model.

    Parameters:
    - model: The neural network model to be trained.
    - data: Preprocessed feature data.
    - labels: Corresponding labels for the data.

    Returns:
    - model: Trained neural network model.
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred_class)
    print('Validation Accuracy:', accuracy)
    print('Validation Classification Report:')
    print(classification_report(y_val, y_pred_class))

    return model
