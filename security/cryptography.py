import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class Cryptography:
    def __init__(self):
        pass

    @staticmethod
    def generate_keys():
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

    @staticmethod
    def sign_transaction(private_key, transaction):
        signer = serialization.load_pem_private_key(
            private_key,
            password=None,
            backend=default_backend()
        )
        signature = signer.sign(
            transaction.encode(),
            padding=serialization_PKCS1v15(),
            algorithm=hashes.SHA256()
        )
        return signature

    @staticmethod
    def verify_signature(public_key, transaction, signature):
        verifier = serialization.load_ssh_public_key(
            public_key,
            backend=default_backend()
        )
        verifier.verify(
            signature,
            transaction.encode(),
            padding=serialization_PKCS1v15(),
            algorithm=hashes.SHA256()
        )
        return True
