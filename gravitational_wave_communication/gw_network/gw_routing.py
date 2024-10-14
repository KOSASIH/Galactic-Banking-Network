# gw_routing.py

import gw_topology

def create_routing_table(nodes):
    # Create a routing table based on the network topology
    routing_table = {}
    for node in nodes:
        routing_table[node.node_id] = {}
        for other_node in nodes:
            if other_node != node:
                path = gw_topology.get_shortest_path(node, other_node)
                routing_table[node.node_id][other_node.node_id] = path
    return routing_table
