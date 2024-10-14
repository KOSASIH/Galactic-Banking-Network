# gw_topology.py

class GWTopology:
    def __init__(self, nodes):
        self.nodes = nodes
        self.adjacency_matrix = self.create_adjacency_matrix()

    def create_adjacency_matrix(self):
        # Create an adjacency matrix representing the network topology
        adjacency_matrix = [[0 for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]
        for i, node in enumerate(self.nodes):
            for j, other_node in enumerate(self.nodes):
                if i != j:
                    adjacency_matrix[i][j] = 1  # assume all nodes are connected
        return adjacency_matrix

    def get_shortest_path(self, source_node, destination_node):
        # Get the shortest path between two nodes in the network
        shortest_path = []
        current_node = source_node
        while current_node != destination_node:
            next_node = self.get_next_node(current_node, destination_node)
            shortest_path.append(next_node)
            current_node = next_node
        return shortest_path

    def get_next_node(self, current_node, destination_node):
        # Get the next node in the shortest path
        for i, node in enumerate(self.nodes):
            if self.adjacency_matrix[self.nodes.index(current_node)][i] == 1:
                if node == destination_node:
                    return node
                else:
                    return node  # assume all nodes are connected
