import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class User:
    def __init__(self, user_id, username, password, email, account):
        self.user_id = user_id
        self.username = username
        self.password = self._hash_password(password)
        self.email = email
        self.account = account
        self.public_key, self.private_key = self._generate_keys()

    def _hash_password(self, password):
        salt = secrets.get("salt")
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return hashed_password

    def _generate_keys(self):
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        private_key = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )
        return public_key, private_key

    def authenticate(self, password):
        hashed_password = self._hash_password(password)
        return hashed_password == self.password

    def get_public_key(self):
        return self.public_key.decode()

    def get_private_key(self):
        return self.private_key.decode()

    def encrypt_data(self, data):
        encryption_key = secrets.get("encryption_key")
        f = Fernet(encryption_key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        encryption_key = secrets.get("encryption_key")
        f = Fernet(encryption_key)
        decrypted_data = f.decrypt(encrypted_data)
        return decrypted_data.decode()
