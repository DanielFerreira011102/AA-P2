import os
import re
from dataclasses import fields

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import use
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

use('TkAgg')


class Graph(nx.Graph):

    def __init__(self, num_vertices: int = None, edge_percentage: float = None, name: str = None, path=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.graph['num_vertices'] = num_vertices
        self.graph['edge_percentage'] = edge_percentage
        self.hash = self._generate_hash_name()
        self.path = path
        self.name = name if name else self.path if self.path else self.hash

    def number_of_vertices(self):
        return self.graph.get('num_vertices', None)

    def edge_percentage(self):
        return self.graph.get('edge_percentage', None)

    def visualize(self, *args, **kwargs):
        plt.figure(kwargs.get('figsize', None))
        nx.draw(self, *args, **kwargs)
        plt.show()

    def save(self, path: str = None):
        """Saves the graph to a file."""

        def validate_path(p):
            b = os.path.basename(p)
            if not b or b.endswith('.'):
                return False
            if not os.path.splitext(b)[1]:
                return False
            if not all(c.isprintable() and c not in {'/', '\\', '?', '%', '*', ':', '|', '"', '<', '>'} for c in b):
                return False
            return True

        def extract_folder_and_extension(p):
            d, f = os.path.split(p)
            n, e = os.path.splitext(f)
            if e in ('.gz', '.bz2'):
                _, e = os.path.splitext(n)
            return d, e.lstrip('.')

        if path is None:
            path = f"{self.name}.txt"

        if not validate_path(path):
            raise ValueError(f"The path {path} is invalid.")

        folder, extension = extract_folder_and_extension(path)

        if not extension:
            raise ValueError(f"The extension of the path {path} is invalid.")

        match extension:
            case 'gml':
                self._save_gml(path)
            case 'txt':
                self._save_txt(path)
            case _:
                raise ValueError(f"The extension {extension} is not supported.")

    def _save_gml(self, path):
        """Saves the graph to a GML file."""
        nx.write_gml(self, path)

    def _save_txt(self, path):
        """Saves the graph to a text file."""
        with open(path, 'w') as file:
            file.write(f"{int(self.is_directed())}\n")
            file.write(f"{int(self.is_weighted())}\n")
            file.write(f"{self.number_of_nodes()}\n")
            file.write(f"{self.number_of_edges()}\n")

            for edge in self.edges(data=True):
                file.write(f"{edge[0]} {edge[1]} {edge[2]['weight']}\n") if self.is_weighted() else file.write(
                    f"{edge[0]} {edge[1]}\n")

    def is_weighted(self):
        """Returns whether the graph is weighted."""
        return nx.is_weighted(self)

    def maximum_number_of_edges(self):
        """Returns the maximum number of edges for the graph."""
        return self.number_of_nodes() * (self.number_of_nodes() - 1) / 2

    @staticmethod
    def _from_txt_file(filename, mode="dwneV"):
        match mode:
            case "dwneV":
                return Graph._from_txt_file_dwneV(filename)
            case "neA":
                return Graph._from_txt_file_neA(filename)
            case "V":
                return Graph._from_txt_file_V(filename)

    @staticmethod
    def _from_txt_file_V(filename):
        """Reads a graph from a TXT file.

         The file format is as follows:
         - The lines contain information about each edge, with each line representing an edge in the format:
           "vertex1 vertex2 [weight]", where vertex1 and vertex2 are node identifiers. The vertex identifiers can be
           integers, floats, or tuples depending on the nature of the graph. If the graph is weighted, the weight should be a
           floating-point number.

         Parameters
         ----------
         filename : str
             The name of the file to read from.

         Returns
         -------
         Graph
             The graph read from the file.
         """

        def validate_vertex(v):
            if isinstance(v, (int, float)):
                return True
            return False

        def calculate_edge_percentage(nv, ne):
            return (ne / ((nv * (nv - 1)) / 2)) * 100

        with open(filename, 'r') as file:
            graph = Graph()

            for line in file:
                vertex1, vertex2 = line.strip().split()
                vertex1 = eval(vertex1)
                vertex2 = eval(vertex2)
                if not validate_vertex(vertex1) or not validate_vertex(vertex2):
                    raise ValueError("Invalid vertex format in the input file.")
                graph.add_edge(vertex1, vertex2)

        graph.graph['num_vertices'] = graph.number_of_nodes()
        graph.graph['edge_percentage'] = calculate_edge_percentage(graph.number_of_nodes(), graph.number_of_edges())
        graph._generate_hash_name()

        return graph

    @staticmethod
    def _from_txt_file_neA(filename):
        """Reads a graph from a TXT file.

         The file format is as follows:
         - The first line contains the number of nodes.
         - The second line contains the number of edges.
         - The subsequent lines contain the adjacency matrix of the graph, with each line representing a row of the matrix.

         Parameters
         ----------
         filename : str
             The name of the file to read from.

         Returns
         -------
         Graph
             The graph read from the file.
         """

        def calculate_edge_percentage(nv, ne):
            return (ne / ((nv * (nv - 1)) / 2)) * 100

        with open(filename, 'r') as file:
            num_nodes = int(file.readline().strip())
            num_edges = int(file.readline().strip())

            graph = Graph(num_vertices=num_nodes, edge_percentage=calculate_edge_percentage(num_nodes, num_edges))

            graph.add_nodes_from(range(num_nodes))

            for i in range(num_nodes):
                for j, value in enumerate(file.readline().strip().split()):
                    if int(value) == 1:
                        graph.add_edge(i, j)

        return graph

    @staticmethod
    def _from_txt_file_dwneV(filename):
        """Reads a graph from a TXT file.

         The file format is as follows:
         - The first line indicates whether the graph is directed or undirected (0 for undirected, 1 for directed).
         - The second line indicates whether the graph is weighted or unweighted (0 for unweighted, 1 for weighted).
         - The third line contains the number of nodes.
         - The fourth line contains the number of edges.
         - The subsequent lines contain information about each edge, with each line representing an edge in the format:
           "vertex1 vertex2 [weight]", where vertex1 and vertex2 are node identifiers. The vertex identifiers can be
           integers, floats, or tuples depending on the nature of the graph. If the graph is weighted, the weight should be a
           floating-point number.

         Parameters
         ----------
         filename : str
             The name of the file to read from.

         Returns
         -------
         Graph
             The graph read from the file.
         """

        def validate_vertex(v):
            if isinstance(v, (int, float)):
                return True
            return False

        def validate_weight(w):
            return isinstance(w, (int, float))

        def calculate_edge_percentage(nv, ne):
            return (ne / ((nv * (nv - 1)) / 2)) * 100

        with open(filename, 'r') as file:
            is_directed = bool(int(file.readline().strip()))
            is_weighted = bool(int(file.readline().strip()))
            num_nodes = int(file.readline().strip())
            num_edges = int(file.readline().strip())

            graph = Graph(num_vertices=num_nodes, edge_percentage=calculate_edge_percentage(num_nodes, num_edges),
                          directed=is_directed)

            graph.add_nodes_from(range(num_nodes))

            for _ in range(num_edges):
                line = re.findall(r'\([^)]*\)|\S+', file.readline().strip())

                if len(line) not in (2, 3):
                    raise ValueError("Invalid edge format in the input file.")

                vertex1 = eval(line[0])
                vertex2 = eval(line[1])

                if not validate_vertex(vertex1) or not validate_vertex(vertex2):
                    raise ValueError("Invalid vertex format in the input file.")

                if is_weighted:
                    weight = eval(line[2]) if len(line) == 3 else 1

                    if not validate_weight(weight):
                        raise ValueError("Invalid weight format in the input file.")

                    graph.add_edge(vertex1, vertex2, weight=weight)
                else:
                    graph.add_edge(vertex1, vertex2)

        return graph

    @staticmethod
    def _from_csv_file(filename):
        """Reads a graph from a CSV file."""

        def calculate_edge_percentage(nv, ne):
            return (ne / ((nv * (nv - 1)) / 2)) * 100

        def validate_vertex(v):
            if isinstance(v, (int, float)):
                return True
            return False

        graph = Graph()
        with open(filename, 'r') as file:
            file.readline()
            for line in file:
                vertex1, vertex2 = line.strip().split(',')
                vertex1 = eval(vertex1)
                vertex2 = eval(vertex2)
                if not validate_vertex(vertex1) or not validate_vertex(vertex2):
                    raise ValueError("Invalid vertex format in the input file.")
                graph.add_edge(vertex1, vertex2)

        graph.graph['num_vertices'] = graph.number_of_nodes()
        graph.graph['edge_percentage'] = calculate_edge_percentage(graph.number_of_nodes(), graph.number_of_edges())
        graph._generate_hash_name()

        return graph

    @staticmethod
    def _from_gml_file(filename):
        """Reads a graph from a GML file."""

        def destringize_to_tuple(s):
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
                x, y = map(int, s.split(","))
                return x, y
            return s

        graph = nx.read_gml(filename, destringizer=destringize_to_tuple)
        graph.__class__ = Graph
        graph._generate_hash_name()

        return graph

    @staticmethod
    def from_file(path, mode="dwneV"):
        """Reads a graph from a file."""

        logging.info(f"Reading graph from file {path}")

        _, extension = os.path.splitext(path)

        if not extension:
            raise ValueError(f"The extension of the path {path} is invalid.")

        if not os.path.isfile(path):
            raise ValueError(f"The path {path} does not exist.")

        match extension:
            case '.gml':
                graph = Graph._from_gml_file(path)
            case '.csv':
                graph = Graph._from_csv_file(path)
            case '.txt' | _:
                graph = Graph._from_txt_file(path, mode=mode)

        graph.path = path
        graph.name = os.path.basename(path)
        return graph

    @staticmethod
    def from_files(*paths, mode="dwneV"):
        graphs = []
        for path in paths:
            if isinstance(path, str):
                graphs.append(Graph.from_file(path, mode=mode))
            elif isinstance(path, (list, tuple)):
                for nested_path in path:
                    if not isinstance(nested_path, str):
                        raise ValueError(f"The path {nested_path} is invalid.")
                    graphs.append(Graph.from_file(nested_path, mode=mode))
            else:
                raise ValueError(f"The path {path} is invalid.")
        return graphs

    def _generate_hash_name(self):
        self.hash = hash(self)

    def __str__(self):
        return f"Graph(name={self.name}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()}, edge_percentage={self.edge_percentage()})"

    def __repr__(self):
        return str(self)


def main():
    def teste1():
        gs = Graph.from_files(['collections/SW_ALGUNS_GRAFOS/SWtinyG.txt', 'collections/SW_ALGUNS_GRAFOS/SWtinyDG.txt'],
                              'collections/SW_ALGUNS_GRAFOS/SWtinyGTup.txt')
        for g in gs:
            print(g)

    def teste2():
        g = Graph.from_file('collections/random_graphs/BD0/Grafo1.txt', mode='neA')
        print(g)
        g.visualize()

    def teste3():
        g = Graph.from_file('collections/git_web_ml/musae_git_edges.csv')
        print(g)

    def teste4():
        g = Graph.from_file('collections/facebook/1684.edges', mode='V')
        print(g)
        g.visualize()

    # teste1()
    # teste2()
    # teste3()
    # teste4()


if __name__ == '__main__':
    main()
