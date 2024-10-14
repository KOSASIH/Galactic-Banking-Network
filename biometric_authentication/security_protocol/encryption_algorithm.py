# encryption_algorithm.py

from cryptography.fernet import Fernet

class EncryptionAlgorithm:
    def __init__(self):
        # Generate a key for encryption and decryption
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data):
        """
        Encrypt data.

        Parameters:
        - data: Data to encrypt (must be bytes).

        Returns:
        - encrypted_data: Encrypted data.
        """
        # Ensure data is in bytes
        if isinstance(data, str):
            data = data.encode()
        encrypted_data = self.cipher.encrypt(data)
        return encrypted_data

    def decrypt(self, encrypted_data):
        """
        Decrypt data.

        Parameters:
        - encrypted_data: Encrypted data to decrypt.

        Returns:
        - decrypted_data: Decrypted data (as string).
        """
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return decrypted_data.decode()

    def get_key(self):
        """
        Get the encryption key.

        Returns:
        - key: Encryption key.
        """
        return self.key
