import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class StellarConsensusProtocol:
    def __init__(self, node_id, private_key, network_config):
        self.node_id = node_id
        self.private_key = private_key
        self.network_config = network_config
        self.blockchain = []

    def generate_block(self, transactions):
        # Generate a new block and add it to the blockchain
        block_data = f"{self.node_id}:{transactions}".encode()
        signature = self.private_key.sign(block_data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        self.blockchain.append({"block_data": block_data, "signature": signature})

    def verify_block(self, block):
        # Verify the authenticity of a block using the node's public key
        public_key = serialization.load_pem_public _key(self.network_config["public_key"], backend=default_backend())
        try:
            public_key.verify(block["signature"], block["block_data"], padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
            return True
        except InvalidSignature:
            return False

    def get_blockchain(self):
        # Return the current state of the blockchain
        return self.blockchain
