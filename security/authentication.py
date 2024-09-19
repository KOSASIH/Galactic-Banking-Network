import os
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class Authentication:
    def __init__(self):
        pass

    @staticmethod
    def generate_password_hash(password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return key

    @staticmethod
    def verify_password(stored_password, provided_password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(provided_password.encode())
        return key == stored_password

    @staticmethod
    def generate_session_key(node_id, password):
        salt = os.urandom(16)
        password_hash = Authentication.generate_password_hash(password, salt)
        session_key = hashlib.sha256((node_id + password).encode()).digest()
        return session_key, salt

    @staticmethod
    def authenticate_node(node_id, password, stored_password, salt):
        if Authentication.verify_password(stored_password, password, salt):
            return Authentication.generate_session_key(node_id, password)[0]
        else:
            return None
