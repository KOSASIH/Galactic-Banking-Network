import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class Account:
    def __init__(self, user_id, account_number, balance=0.0, currency="Galactic Credits"):
        self.user_id = user_id
        self.account_number = account_number
        self.balance = balance
        self.currency = currency
        self.encrypted_balance = self._encrypt_balance()
        self.public_key, self.private_key = self._generate_keys()

    def _encrypt_balance(self):
        encryption_key = secrets.get("encryption_key")
        f = Fernet(encryption_key)
        encrypted_balance = f.encrypt(str(self.balance).encode())
        return encrypted_balance

    def _decrypt_balance(self):
        encryption_key = secrets.get("encryption_key")
        f = Fernet(encryption_key)
        decrypted_balance = f.decrypt(self.encrypted_balance)
        return float(decrypted_balance.decode())

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

    def deposit(self, amount):
        self.balance += amount
        self.encrypted_balance = self._encrypt_balance()

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient balance")
        self.balance -= amount
        self.encrypted_balance = self._encrypt_balance()

    def get_public_key(self):
        return self.public_key.decode()

    def get_private_key(self):
        return self.private_key.decode()

    def sign_transaction(self, transaction_data):
        private_key = serialization.load_pem_private_key(
            self.private_key,
            password=None,
            backend=default_backend()
        )
        signature = private_key.sign(
            transaction_data.encode(),
            padding=serialization.pkcs7,
            algorithm=hashes.SHA256()
        )
        return signature

    def verify_signature(self, transaction_data, signature):
        public_key = serialization.load_ssh_public_key(
            self.public_key,
            backend=default_backend()
        )
        public_key.verify(
            signature,
            transaction_data.encode(),
            padding=serialization.pkcs7,
            algorithm=hashes.SHA256()
        )
