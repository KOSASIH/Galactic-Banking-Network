# config.py

class Config:
    def __init__(self):
        self.network_nodes = 10  # number of nodes in the network
        self.signal_sensitivity = 0.1  # sensitivity for signal detection
        self.power_level = 0.5  # power level for signal modulation
        self.authentication_key = "secret_key"  # authentication key for nodes
        self.authorization_key = "secret_key"  # authorization key for nodes
