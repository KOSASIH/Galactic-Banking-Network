import hashlib
import hmac
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.x509 import load_pem_x509_certificate
from cryptography.x509.oid import NameOID

class GWCryptography:
    def __init__(self, private_key, public_key):
        self.private_key = serialization.load_pem_private_key(
            private_key,
            password=None,
            backend=default_backend()
        )
        self.public_key = serialization.load_pem_public_key(
            public_key,
            backend=default_backend()
        )

    def generate_key_pair(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def encrypt_data(self, data):
        cipher = Cipher(algorithms.AES(self.private_key), modes.CBC(b'\0' * 16), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        return encryptor.update(padded_data) + encryptor.finalize()

    def decrypt_data(self, encrypted_data):
        cipher = Cipher(algorithms.AES(self.private_key), modes.CBC(b'\0' * 16), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(decrypted_padded_data) + unpadder.finalize()

    def sign_data(self, data):
        signer = self.private_key.signer(
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        signer.update(data)
        return signer.finalize()

    def verify_signature(self, data, signature):
        verifier = self.public_key.verifier(
            signature,
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        verifier.update(data)
        verifier.verify()

    def hash_data(self, data):
        return hashlib.sha256(data).digest()

    def hmac_data(self, data, key):
        return hmac.new(key, data, hashlib.sha256).digest()

def main():
    gw_cryptography = GWCryptography(private_key=open('private_key.pem', 'rb').read(), public_key=open('public_key.pem', 'rb').read())

    data = b'Hello, World!'
    encrypted_data = gw_cryptography.encrypt_data(data)
    print('Encrypted Data:', encrypted_data)

    decrypted_data = gw_cryptography.decrypt_data(encrypted_data)
    print('Decrypted Data:', decrypted_data)

    signature = gw_cryptography.sign_data(data)
    print('Signature:', signature)

    gw_cryptography.verify_signature(data, signature)

    hashed_data = gw_cryptography.hash_data(data)
    print('Hashed Data:', hashed_data)

    hmac_data = gw_cryptography.hmac_data(data, b'secret_key')
    print('HMAC Data:', hmac_data)

if __name__ == '__main__':
    main()
