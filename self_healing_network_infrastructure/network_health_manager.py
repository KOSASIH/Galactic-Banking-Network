# network_health_manager.py

import network
import utils

class NetworkHealthManager:
    def __init__(self, network):
        self.network = network

    def check_network_health(self):
        # Check the overall health of the network
        damaged_nodes = [node for node in self.network.nodes if node.is_damaged()]
        if damaged_nodes:
            return False
        return True
