# gw_network.py

import gw_node
import gw_routing
import gw_topology

class GWNework:
    def __init__(self, nodes):
        self.nodes = nodes
        self.routing_table = gw_routing.create_routing_table(nodes)
        self.topology = gw_topology.GWTopology(nodes)

    def send_signal(self, source_node, destination_node, signal):
        # Send gravitational wave signal from source node to destination node
        path = self.topology.get_shortest_path(source_node, destination_node)
        for node in path:
            node.receive_signal(signal)
        return signal

    def receive_signal(self, node, signal):
        # Receive gravitational wave signal at a node
        node.process_signal(signal)
        return signal

    def get_node(self, node_id):
        # Get a node by its ID
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_nodes(self):
        # Get all nodes in the network
        return self.nodes
