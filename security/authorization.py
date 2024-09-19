import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class Authorization:
    def __init__(self):
        self.roles = {
            'admin': ['create_account', 'delete_account', 'transfer_funds'],
            'user': ['view_account', 'transfer_funds']
        }
        self.node_attributes = {}
        self.role_hierarchy = {
            'admin': ['user'],
            'user': []
        }
        self.permission_matrix = {
            'create_account': ['admin'],
            'delete_account': ['admin'],
            'transfer_funds': ['admin', 'user'],
            'view_account': ['user']
        }

    def generate_node_attributes(self, node_id, location, time_of_day):
        attributes = {
            'location': location,
            'time_of_day': time_of_day
        }
        self.node_attributes[node_id] = attributes

    def is_authorized(self, node_id, role, action):
        if role in self.roles:
            if action in self.roles[role]:
                return True
            elif self.is_inherited_role(role, action):
                return True
            elif self.is_attribute_based_authorized(node_id, action):
                return True
        return False

    def is_inherited_role(self, role, action):
        for parent_role in self.role_hierarchy[role]:
            if action in self.roles[parent_role]:
                return True
        return False

    def is_attribute_based_authorized(self, node_id, action):
        attributes = self.node_attributes[node_id]
        for permission, required_attributes in self.permission_matrix[action].items():
            if all(attributes[attr] == required_attributes[attr] for attr in required_attributes):
                return True
        return False

    def authorize_node(self, node_id, role, action):
        if self.is_authorized(node_id, role, action):
            return True
        else:
            return False

    def generate_permission_token(self, node_id, role, action):
        if self.authorize_node(node_id, role, action):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            token = private_key.sign(
                f"{node_id}:{role}:{action}".encode(),
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return token
        else:
            return None

    def verify_permission_token(self, token, node_id, role, action):
        public_key = serialization.load_pem_public_key(
            f"node_{node_id}_public_key.pem",
            backend=default_backend()
        )
        try:
            public_key.verify(
                token,
                f"{node_id}:{role}:{action}".encode(),
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except ValueError:
            return False
