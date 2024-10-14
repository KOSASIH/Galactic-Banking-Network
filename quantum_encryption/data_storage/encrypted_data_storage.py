# encrypted_data_storage.py

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import depolarizing_error, NoiseModel
from qiskit.tools.monitor import job_monitor
from quantum_encryption import quantum_encrypt

class EncryptedDataStorage:
    def __init__(self, error_rate=0.01):
        self.error_rate = error_rate
        self.encrypted_data = {}

    def store_data(self, data, key):
        """
        Store data in the encrypted data storage system.

        Parameters:
        - data: Data to be stored.
        - key: Quantum key.
        """
        encrypted_data = quantum_encrypt(data, key, self.error_rate)
        self.encrypted_data[key] = encrypted_data

    def retrieve_data(self, key):
        """
        Retrieve data from the encrypted data storage system.

        Parameters:
        - key: Quantum key.

        Returns:
        - encrypted_data: Encrypted data.
        """
        return self.encrypted_data[key]

    def delete_data(self, key):
        """
        Delete data from the encrypted data storage system.

        Parameters:
        - key: Quantum key.
        """
        del self.encrypted_data[key]
