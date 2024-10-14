# gw_authorization.py

import gw_authentication

class GWAuthorization:
    def __init__(self, authorization_key):
        self.authorization_key = authorization_key
        self.authorized_nodes = {}

    def authorize_node(self, node_id, authorization_data):
        # Authorize a node using a cryptographic key
        expected_data = gw_cryptography.encrypt_signal(node_id.encode(), self.authorization_key)
        if authorization_data == expected_data:
            self.authorized_nodes[node_id] = True
            return True
        return False

    def is_node_authorized(self, node_id):
        # Check if a node is authorized
        return node_id in self.authorized_nodes
