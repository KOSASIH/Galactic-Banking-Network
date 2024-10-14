# biometric_auth_utils.py

import hashlib

def hash_biometric_data(data):
    """
    Hash biometric data for secure storage.

    Parameters:
    - data: Biometric data to hash (as string).

    Returns:
    - hashed_data: Hashed biometric data.
    """
    # Use a secure hashing algorithm to hash the biometric data
    hashed_data = hashlib.sha256(data.encode()).hexdigest()
    return hashed_data

def compare_biometric_data(data1, data2):
    """
    Compare two biometric data samples for authentication.

    Parameters:
    - data1: First biometric data sample (as string).
    - data2: Second biometric data sample (as string).

    Returns:
    - match: True if the biometric data samples match, False otherwise.
    """
    # Use a secure comparison algorithm to compare the biometric data samples
    # For demonstration purposes, we'll just use a simple string comparison
    return data1 == data2
