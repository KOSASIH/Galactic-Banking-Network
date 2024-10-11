import requests
import json

class GalacticBlockchainExplorer:
    def __init__(self, node_id, private_key, network_config):
        self.node_id = node_id
        self.private_key = private_key
        self.network_config = network_config
        self.base_url = f"https://{self.network_config['blockchain_explorer']}/api"

    def get_blockchain_info(self):
        # Get blockchain info
        response = requests.get(f"{self.base_url}/blockchain/info")
        return response.json()

    def get_block(self, block_height):
        # Get block by height
        response = requests.get(f"{self.base_url}/block/{block_height}")
        return response.json()

    def get_transaction(self, transaction_id):
        # Get transaction by ID
        response = requests.get(f"{self.base_url}/transaction/{transaction_id}")
        return response.json()

    def get_address(self, address):
        # Get address info
        response = requests.get(f"{self.base_url}/address/{address}")
        return response.json()

    def get_blockchain_stats(self):
        # Get blockchain stats
        response = requests.get(f"{self.base_url}/blockchain/stats")
        return response.json()

def main():
    node_id = "node1"
    private_key = "private_key"
    network_config = {
        "blockchain_explorer": "galactic-explorer.com"
    }

    galactic_blockchain_explorer = GalacticBlockchainExplorer(node_id, private_key, network_config)
    print("Blockchain Info:")
    print(galactic_blockchain_explorer.get_blockchain_info())
    print("Block 100:")
    print(galactic_blockchain_explorer.get_block(100))
    print("Transaction 123456:")
    print(galactic_blockchain_explorer.get_transaction("123456"))
    print("Address 0x123456:")
    print(galactic_blockchain_explorer.get_address("0x123456"))
    print("Blockchain Stats:")
    print(galactic_blockchain_explorer.get_blockchain_stats())

if __name__ == "__main__":
    main()
