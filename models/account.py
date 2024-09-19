import hashlib
from cryptography.fernet import Fernet

class Account:
    def __init__(self, user_id, account_number, balance=0.0):
        self.user_id = user_id
        self.account_number = account_number
        self.balance = balance
        self.encrypted_balance = self._encrypt_balance()

    def _encrypt_balance(self):
        encryption_key = secrets.get("encryption_key")
        f = Fernet(encryption_key)
        encrypted_balance = f.encrypt(str(self.balance).encode())
        return encrypted_balance

    def deposit(self, amount):
        self.balance += amount
        self.encrypted_balance = self._encrypt_balance()

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient balance")
        self.balance -= amount
        self.encrypted_balance = self._encrypt_balance()
