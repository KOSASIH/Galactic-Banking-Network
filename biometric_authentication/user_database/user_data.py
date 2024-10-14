# user_data.py

from security_protocol.encryption_algorithm import EncryptionAlgorithm

class UserData:
    def __init__(self):
        # Initialize the encryption algorithm
        self.encryption_algorithm = EncryptionAlgorithm()

    def encrypt_user_data(self, user_data):
        """
        Encrypt user data.

        Parameters:
        - user_data: User data to encrypt (must be bytes or string).

        Returns:
        - encrypted_user_data: Encrypted user data.
        """
        # Ensure user data is in bytes or string
        if isinstance(user_data, dict):
            user_data = str(user_data)
        encrypted_user_data = self.encryption_algorithm.encrypt(user_data)
        return encrypted_user_data

    def decrypt_user_data(self, encrypted_user_data):
        """
        Decrypt user data.

        Parameters:
        - encrypted_user_data: Encrypted user data to decrypt.

        Returns:
        - decrypted_user_data: Decrypted user data (as string).
        """
        decrypted_user_data = self.encryption_algorithm.decrypt(encrypted_user_data)
        return decrypted_user_data

    def store_user_data(self, user_data):
        """
        Store user data securely.

        Parameters:
        - user_data: User data to store (must be bytes or string).

        Returns:
        - stored_user_data: Stored user data.
        """
        # Encrypt user data before storing
        encrypted_user_data = self.encrypt_user_data(user_data)
        # Store encrypted user data (e.g., in a database or file)
        # For demonstration purposes, we'll just return the encrypted data
        return encrypted_user_data

    def retrieve_user_data(self, encrypted_user_data):
        """
        Retrieve and decrypt stored user data.

        Parameters:
        - encrypted_user_data: Encrypted user data to retrieve and decrypt.

        Returns:
        - decrypted_user_data: Decrypted user data (as string).
        """
        # Decrypt stored user data
        decrypted_user_data = self.decrypt_user_data(encrypted_user_data)
        return decrypted_user_data
