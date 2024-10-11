import hashlib
import hmac
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class GalacticBlockchainSecurity:
    def __init__(self, node_id, private_key, network_config):
        self.node_id = node_id
        self.private_key = private_key
        self.network_config = network_config

    def generate_key_pair(self):
        # Generate key pair
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
        return private_key, public_key

    def sign_transaction(self, transaction, private_key):
        # Sign transaction
        signer = hmac.new(private_key, digestmod=hashlib.sha256)
        signer.update(transaction.encode())
        signature = base64.b64encode(signer.digest())
        return signature

    def verify_signature(self, transaction, signature, public_key):
        # Verify signature
        verifier = hmac.new(public_key, digestmod=hashlib.sha256)
        verifier.update(transaction.encode())
        expected_signature = base64.b64encode(verifier.digest())
        return hmac.compare_digest(signature, expected_signature)

def main():
    node_id = "node1"
    private_key = "private_key"
    network_config = {
        "blockchain_explorer": "galactic-explorer.com"
    }

    galactic_blockchain_security = GalacticBlockchainSecurity(node_id, private_key, network_config)
    private_key, public_key = galactic_blockchain_security.generate_key_pair()
    print("Private Key:")
    print(private_key.decode())
    print("Public Key:")
    print(public_key.decode())

    transaction = "Hello, Galactic Blockchain!"
    signature = galactic_blockchain_security.sign_transaction(transaction, private_key)
    print("Signature:")
    print(signature.decode())

    is_valid = galactic_blockchain_security.verify_signature(transaction, signature, public_key)
    print("Is Valid:", is_valid)

if __name__ == "__main__":
    main()
