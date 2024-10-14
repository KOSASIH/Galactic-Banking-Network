# user_database.py

from user_data.user_data import UserData

class UserDatabase:
    def __init__(self):
        # Initialize the user data handler
        self.user_data_handler = UserData()

    def add_user(self, user_data):
        """
        Add a new user to the database.

        Parameters:
        - user_data: User data to add (must be bytes or string).

        Returns:
        - stored_user_data: Stored user data.
        """
        # Encrypt user data before storing
        encrypted_user_data = self.user_data_handler.encrypt_user_data(user_data)
        # Store encrypted user data in the database
        # For demonstration purposes, we'll just store it in a dictionary
        self.database = {}
        self.database[user_data] = encrypted_user_data
        return encrypted_user_data

    def get_user(self, user_id):
        """
        Retrieve a user from the database.

        Parameters:
        - user_id: ID of the user to retrieve.

        Returns:
        - user_data: Retrieved user data (as string).
        """
        # Retrieve encrypted user data from the database
        encrypted_user_data = self.database.get(user_id)
        if encrypted_user_data:
            # Decrypt stored user data
            decrypted_user_data = self.user_data_handler.decrypt_user_data(encrypted_user_data)
            return decrypted_user_data
        else:
            return None

    def update_user(self, user_id, updated_user_data):
        """
        Update an existing user in the database.

        Parameters:
        - user_id: ID of the user to update.
        - updated_user_data: Updated user data (must be bytes or string).

        Returns:
        - updated_user_data: Updated user data.
        """
        # Retrieve encrypted user data from the database
        encrypted_user_data = self.database.get(user_id)
        if encrypted_user_data:
            # Decrypt stored user data
            decrypted_user_data = self.user_data_handler.decrypt_user_data(encrypted_user_data)
            # Update user data
            updated_user_data = self.user_data_handler.encrypt_user_data(updated_user_data)
            self.database[user_id] = updated_user_data
            return updated_user_data
        else:
            return None

    def delete_user(self, user_id):
        """
        Delete a user from the database.

        Parameters:
        - user_id: ID of the user to delete.

        Returns:
        - deleted_user_data: Deleted user data (as string).
        """
        # Retrieve encrypted user data from the database
        encrypted_user_data = self.database.get(user_id)
        if encrypted_user_data:
            # Decrypt stored user data
            decrypted_user_data = self.user_data_handler.decrypt_user_data(encrypted_user_data)
            # Delete user data from the database
            del self.database[user_id]
            return decrypted_user_data
        else:
            return None
