import base64
import json
import zlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class GWDataEncoding:
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

    def encode_data(self, data):
        json_data = json.dumps(data)
        compressed_data = zlib.compress(json_data.encode('utf-8'))
        encoded_data = base64.b64encode(compressed_data)
        return encoded_data

    def decode_data(self, encoded_data):
        decoded_data = base64.b64decode(encoded_data)
        decompressed_data = zlib.decompress(decoded_data)
        json_data = decompressed_data.decode('utf-8')
        return json.loads(json_data)

    def encrypt_data(self, data):
        encrypted_data = self.private_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data

def main():
    gw_data_encoding = GWDataEncoding(private_key=open('private_key.pem', 'rb').read(), public_key=open('public_key.pem', 'rb').read())

    data = {'message': 'Hello, World!'}
    encoded_data = gw_data_encoding.encode_data(data)
    print('Encoded Data:', encoded_data)

    decoded_data = gw_data_encoding.decode_data(encoded_data)
    print('Decoded Data:', decoded_data)

    encrypted_data = gw_data_encoding.encrypt_data(encoded_data)
    print('Encrypted Data:', encrypted_data)

    decrypted_data = gw_data_encoding.decrypt_data(encrypted_data)
    print('Decrypted Data:', decrypted_data)

if __name__ == '__main__':
    main()
