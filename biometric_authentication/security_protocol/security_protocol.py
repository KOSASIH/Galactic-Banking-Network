# security_protocol.py

import hashlib

class SecurityProtocol:
    def __init__(self, encryption_algorithm):
        self.encryption_algorithm = encryption_algorithm

    def encrypt_data(self, data):
        """
        Encrypt data.

        Parameters:
        - data: Data to encrypt.

        Returns:
        - encrypted_data: Encrypted data.
        """
        # Implement data encryption algorithm here
        return self.encryption_algorithm.encrypt(data)

    def decrypt_data(self, encrypted_data):
        """
        Decrypt data.

        Parameters:
        - encrypted_data: Encrypted data to decrypt.

        Returns:
        - decrypted_data: Decrypted data.
        """
        # Implement data decryption algorithm here
        return self.encryption_algorithm.decrypt(encrypted_data)

    def hash_data(self, data):
        """
        Hash data.

        Parameters:
        - data: Data to hash.

        Returns:
        - hashed_data: Hashed data.
        """
        # Implement data hashing algorithm here
        return hashlib.sha256(data).hexdigest()
