# neuro_interface_utils.py

import numpy as np

def process_neural_signals(signals):
    """
    Process neural signals from the neuro interface.

    Parameters:
    - signals: Neural signals to process (as numpy array).

    Returns:
    - processed_signals: Processed neural signals.
    """
    # Apply filters to remove noise and artifacts
    filtered_signals = np.array([signal * 0.5 for signal in signals])
    return filtered_signals

def extract_features(signals):
    """
    Extract features from the processed neural signals.

    Parameters:
    - signals: Processed neural signals (as numpy array).

    Returns:
    - features: Extracted features.
    """
    # Calculate mean and standard deviation of the signals
    mean = np.mean(signals)
    std_dev = np.std(signals)
    return [mean, std_dev]

def classify_brain_state(features):
    """
    Classify the brain state based on the extracted features.

    Parameters:
    - features: Extracted features.

    Returns:
    - brain_state: Classified brain state.
    """
    # Use a machine learning model to classify the brain state
    # For demonstration purposes, we'll just use a simple threshold
    if features[0] > 0.5:
        return "active"
    else:
        return "idle"
