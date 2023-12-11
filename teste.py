import networkx as nx
import matplotlib.pyplot as plt

# Create an undirected graph
g = nx.Graph()

# Add edges with weights
g.add_edges_from([(1, 2, {'weight': 3}),
                  (1, 3, {'weight': 50}),
                  (2, 3, {'weight': 1}),
                  (2, 4, {'weight': 6}),
                  (3, 4, {'weight': 4})])

# Draw the graph
pos = nx.spring_layout(g)  # You can choose a different layout if needed
nx.draw(g, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold")

# Add edge labels (costs)
edge_labels = nx.get_edge_attributes(g, 'weight')
nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

# Show the graph
plt.show()



class Graph:
    def __init__(self):
        self.nodes = set()
        self.visited = set()
        self.edges = {}
        self.beacons = set()

    def add_edges_from(self, edges):
        for node1, node2, cost in edges:
            self.add_edge(node1, node2, cost)

    def add_edge(self, node1, node2, cost):
        self.nodes.add(node1)
        self.nodes.add(node2)
        self.edges.setdefault(node1, {})[node2] = cost
        self.edges.setdefault(node2, {})[node1] = cost

    def set_visited(self, node):
        self.visited.add(node)

    def unknown_nodes(self):
        return self.nodes.difference(self.visited)

    def cost(self, node1, node2):
        return self.edges[node1].get(node2, float('inf'))

    def find_best_path(self, start_node):
        def dfs(current_node, remaining_nodes, visited_nodes):
            if not remaining_nodes:
                return [current_node], 0

            best_path = None
            min_cost = float('inf')

            for next_node in remaining_nodes:
                if next_node not in visited_nodes:
                    new_path, new_cost = dfs(
                        next_node, remaining_nodes - {next_node}, visited_nodes | {next_node}
                    )
                    total_cost = self.cost(current_node, next_node) + new_cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_path = [current_node] + new_path

            return best_path, min_cost

        unknown_nodes = self.unknown_nodes()
        return dfs(start_node, unknown_nodes - {start_node}, {start_node})



g = Graph()
g.add_edges_from([(1, 2, 3), (1, 3, 50), (2, 3, 1), (2, 4, 6), (3, 4, 4)])
g.set_visited(2)
start_node = 2
path, cost = g.find_best_path(start_node)
print("Path:", path) # (2, 1, 2, 3, 4)
print("Cost:", cost) # 11

"""
Example:
start -> 2 | {2}
go to 1, cost = 3 | (1, 2, 3)
1 | {2, 1}
go to 2, cost = 3 + 3 = 6 | (1, 2, 3)
2 | {2, 1, 2}
go to 3, cost = 1  + 6 = 7 | (2, 3, 1)
3 | {2, 1, 2, 3}
go to 4, cost = 4 + 7 = 11 | (3, 4, 4)
4 | {2, 1, 2, 3, 4}
return [(2, 1, 2, 3, 4), 11]
"""
