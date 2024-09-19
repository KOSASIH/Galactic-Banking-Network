import socket
import pickle
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class Communication:
    def __init__(self, node):
        self.node = node
        self.encryption_key = secrets.get("encryption_key")

    def send_message(self, message, peer_id):
        connection = self.node.peers.get(peer_id)
        if connection:
            encrypted_message = self._encrypt_message(message)
            connection.send(pickle.dumps(encrypted_message))
        else:
            print(f"Peer {peer_id} not connected")

    def broadcast_message(self, message):
        for peer_id, connection in self.node.peers.items():
            encrypted_message = self._encrypt_message(message)
            connection.send(pickle.dumps(encrypted_message))

    def _encrypt_message(self, message):
        f = Fernet(self.encryption_key)
        encrypted_message = f.encrypt(pickle.dumps(message))
        return {"type": "encrypted", "message": encrypted_message}

    def receive_message(self, data):
        message = pickle.loads(data)
        if message["type"] == "encrypted":
            decrypted_message = self._decrypt_message(message["message"])
            return decrypted_message
        else:
            return message

    def _decrypt_message(self, encrypted_message):
        f = Fernet(self.encryption_key)
        decrypted_message = f.decrypt(encrypted_message)
        return pickle.loads(decrypted_message)
