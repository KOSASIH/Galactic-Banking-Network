import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class Transaction:
    def __init__(self, sender_account, recipient_account, amount, currency="Galactic Credits"):
        self.sender_account = sender_account
        self.recipient_account = recipient_account
        self.amount = amount
        self.currency = currency
        self.timestamp = datetime.datetime.now()
        self.transaction_id = self._generate_transaction_id()
        self.signature = self._sign_transaction()

    def _generate_transaction_id(self):
        transaction_id = hashlib.sha256(str(self.timestamp).encode()).hexdigest()
        return transaction_id

    def _sign_transaction(self):
        transaction_data = f"{self.sender_account.user_id}{self.recipient_account.user_id}{self.amount}{self.currency}{self.timestamp}"
        signature = self.sender_account.sign_transaction(transaction_data)
        return signature

    def verify_signature(self):
        transaction_data = f"{self.sender_account.user_id}{self.recipient_account.user_id}{self.amount}{self.currency}{self.timestamp}"
        self.sender_account.verify_signature(transaction_data, self.signature)

    def execute_transaction(self):
        if self.verify_signature():
            self.sender_account.withdraw(self.amount)
            self.recipient_account.deposit(self.amount)
            return True
        else:
            raise ValueError("Invalid transaction signature")
            return False

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "sender_account": self.sender_account.user_id,
            "recipient_account": self.recipient_account.user_id,
            "amount": self.amount,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat()
        }
