import socket
import threading
import pickle
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class Node:
    def __init__(self, node_id, host, port, network):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.network = network
        self.peers = {}
        self.transactions = {}
        self.public_key, self.private_key = self._generate_keys()

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

    def start_node(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        print(f"Node {self.node_id} started on {self.host}:{self.port}")

        threading.Thread(target=self._listen_for_peers).start()

    def _listen_for_peers(self):
        while True:
            connection, address = self.socket.accept()
            threading.Thread(target=self._handle_peer, args=(connection, address)).start()

    def _handle_peer(self, connection, address):
        peer_id = connection.recv(1024).decode()
        self.peers[peer_id] = connection
        print(f"Connected to peer {peer_id}")

        while True:
            data = connection.recv(1024)
            if not data:
                break
            self._handle_message(data)

    def _handle_message(self, data):
        message = pickle.loads(data)
        if message["type"] == "transaction":
            self._handle_transaction(message["transaction"])
        elif message["type"] == "block":
            self._handle_block(message["block"])

    def _handle_transaction(self, transaction):
        if transaction.verify_signature():
            self.transactions[transaction.transaction_id] = transaction
            print(f"Received valid transaction {transaction.transaction_id}")
        else:
            print(f"Received invalid transaction {transaction.transaction_id}")

    def _handle_block(self, block):
        # TO DO: implement block handling logic
        pass

    def broadcast_transaction(self, transaction):
        for peer_id, connection in self.peers.items():
            connection.send(pickle.dumps({"type": "transaction", "transaction": transaction}))

    def broadcast_block(self, block):
        for peer_id, connection in self.peers.items():
            connection.send(pickle.dumps({"type": "block", "block": block}))
