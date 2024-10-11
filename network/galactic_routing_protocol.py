import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class GalacticRoutingProtocol:
    def __init__(self, node_id, private_key):
        self.node_id = node_id
        self.private_key = private_key
        self.routing_table = {}

    def generate_routing_token(self, destination_node):
        # Generate a routing token using the node's private key and destination node ID
        token_data = f"{self.node_id}:{destination_node}".encode()
        signature = self.private_key.sign(token_data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return signature

    def verify_routing_token(self, token, source_node):
        # Verify the routing token using the source node's public key
        public_key = serialization.load_pem_public_key(source_node.public_key, backend=default_backend())
        try:
            public_key.verify(token, f"{source_node.node_id}:{self.node_id}".encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
            return True
        except InvalidSignature:
            return False

    def update_routing_table(self, node_id, public_key):
        # Update the routing table with the node's public key
        self.routing_table[node_id] = public_key
