import json
import cryptography.fernet

class Communication:
    def __init__(self, node):
        self.node = node
        self.encryption_key = secrets.get("encryption_key")
        self.f = cryptography.fernet.Fernet(self.encryption_key)

    def send_message(self, message):
        encrypted_message = self.f.encrypt(json.dumps(message).encode())
        self.node.socket.sendall(encrypted_message)

    def receive_message(self):
        encrypted_message = self.node.socket.recv(1024)
        decrypted_message = self.f.decrypt(encrypted_message)
        return json.loads(decrypted_message.decode())
