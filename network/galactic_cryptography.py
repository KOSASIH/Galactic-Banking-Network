import hashlib
import hmac
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes

class GalacticCryptography:
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

    def encrypt_data(self, data, public_key):
        # Encrypt data
        public_key = serialization.load_ssh_public_key(public_key, backend=default_backend())
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data

    def decrypt_data(self, encrypted_data, private_key):
        # Decrypt data
        private_key = serialization.load_pem_private_key(private_key, password=None, backend=default_backend())
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data

    def sign_data(self, data, private_key):
        # Sign data
        private_key = serialization.load_pem_private_key(private_key, password=None, backend=default_backend())
        signer = private_key.signer(
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        signature = signer.sign(data)
        return signature

    def verify_signature(self, data, signature, public_key):
        # Verify signature
        public_key = serialization.load_ssh_public_key(public_key, backend=default_backend())
        verifier = public_key.verifier(
            signature,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        verifier.update(data)
        verifier.verify()

    def encrypt_symmetric(self, data, key):
        # Encrypt symmetric
        cipher = Cipher(algorithms.AES(key), modes.CBC(b'\00' * 16), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()
        return encrypted_data

    def decrypt_symmetric(self, encrypted_data, key):
        # Decrypt symmetric
        cipher = Cipher(algorithms.AES(key), modes.CBC(b'\00' * 16), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
        return decrypted_data

def main():
    node_id = "node1"
    private_key = "private_key"
    network_config = {
        "blockchain_explorer": "galactic-explorer.com"
    }

    galactic_cryptography = GalacticCryptography(node_id, private_key, network_config)
    private_key, public_key = galactic_cryptography.generate_key_pair()
    print("Private Key:")
    print(private_key.decode())
    print("Public Key:")
    print(public_key.decode())

    data = b"Hello, Galactic Cryptography!"
    encrypted_data = galactic_cryptography.encrypt_data(data, public_key)
    print("Encrypted Data:")
    print(encrypted_data.hex())

    decrypted_data = galactic_cryptography.decrypt_data(encrypted_data, private_key)
    print("Decrypted Data:")
    print(decrypted_data.decode())

    signature = galactic_cryptography.sign_data(data, private_key)
    print("Signature:")
    print(signature.hex())

    galactic_cryptography.verify_signature(data, signature, public_key)
    print("Signature is valid!")

    symmetric_key = b'\00' * 32
    encrypted_data = galactic_cryptography.encrypt_symmetric(data, symmetric_key)
    print("Encrypted Data (Symmetric):")
    print(encrypted_data.hex())

    decrypted _data = galactic_cryptography.decrypt_symmetric(encrypted_data, symmetric_key)
    print("Decrypted Data (Symmetric):")
    print(decrypted_data.decode())

if __name__ == "__main__":
    main()
