import argparse
import atexit
import glob
import os
import random
import re
from math import log, comb
from typing import Union, List
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import pandas as pd
from tqdm import tqdm

from graph import Graph
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


class Solution:
    def __init__(self, graph: Graph, k, kn, success_count, success, vertices, total_time, average_repetition_time,
                 success_probability, min_success_iteration, max_success_iteration, average_success_iteration,
                 average_iterations, iterations, iterations_percentage, repetitions, average_operations_count):
        self.graph = graph
        self.k = k
        self.kn = kn
        self.success_count = success_count
        self.success = success
        self.vertices = vertices
        self.total_time = total_time
        self.min_success_iteration = min_success_iteration
        self.max_success_iteration = max_success_iteration
        self.average_success_iteration = average_success_iteration
        self.average_iterations = average_iterations
        self.average_repetition_time = average_repetition_time
        self.success_probability = success_probability
        self.iterations = iterations
        self.iterations_percentage = iterations_percentage
        self.repetitions = repetitions
        self.average_operations_count = average_operations_count

    def __str__(self):
        return f"Solution(" \
                f"graph={self.graph.name}, " \
                f"k={self.k}, " \
                f"kn={self.kn}, " \
                f"success={self.success}, " \
                f"success_count={self.success_count}, " \
                f"vertices={self.vertices}, " \
                f"total_time={self.total_time}, " \
                f"average_repetition_time={self.average_repetition_time}, " \
                f"min_success_iteration={self.min_success_iteration}, " \
                f"max_success_iteration={self.max_success_iteration}, " \
                f"average_success_iteration={self.average_success_iteration}, " \
                f"average_iterations={self.average_iterations}, " \
                f"success_probability={self.success_probability}, " \
                f"iterations={self.iterations}, " \
                f"iterations_percentage={self.iterations_percentage}, " \
                f"repetitions={self.repetitions}, " \
                f"average_operations_count={self.average_operations_count}" \
                f")"


    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.graph == other.graph and self.vertices == other.vertices

    def __hash__(self):
        return hash((self.graph, self.vertices))

    def __bool__(self):
        return self.success

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __contains__(self, vertex):
        return vertex in self.vertices

    def __getitem__(self, index):
        return self.vertices[index]

    def visualize(self):
        plt.figure(figsize=(10, 5))

        plt.title(f"Solution for {self.graph.name} with k={self.k}")

        plt.axis('on')
        plt.grid(True)

        pos = nx.spring_layout(self.graph, seed=39299899)

        nx.draw(self.graph, pos, node_size=100,
                node_color=["tab:red" if self.success and node in self.vertices else "tab:blue" for node
                            in self.graph.nodes()], edge_color="black", width=1)

        plt.tight_layout()
        plt.show()

    def table(self):
        print(
            f"{'Graph':<25}{'Number of Nodes':<30}{'Edge Percentage':<30}{'Number of Edges':<30}"
            f"{'k':<20}{'kn':<20}{'Iterations':<20}{'Iterations Percentage':<30}{'Repetitions':<20}"
            f"{'Total Time':<20}{'Average Repetition Time':<30}{'Success Count':<20}"
            f"{'Success Probability':<25}{'Min Success Iteration':<30}{'Max Success Iteration':<30}"
            f"{'Average Success Iteration':<30}{'Average Iterations':<30}{'Operations Count':<30}{'Success':<20}{'Vertices':<20}")

        print(
            f"{self.graph.name:<25}{self.graph.number_of_nodes():<30}{self.graph.edge_percentage():<30}"
            f"{self.graph.number_of_edges():<30}{self.k:<20}{self.kn:<20}{self.iterations:<20}{self.iterations_percentage:<30}{self.repetitions:<20}"
            f"{self.total_time:<20.4f}{self.average_repetition_time:<30.4f}{self.success_count:<20}"
            f"{self.success_probability:<25.4f}{self.min_success_iteration:<30}{self.max_success_iteration:<30}"
            f"{self.average_success_iteration:<30.4f}{self.average_iterations:<30.4f}{self.average_operations_count:<30.4f}{self.success:<20}{str(self.vertices):<20}")
        print()


class ISDPSolver:
    def __init__(self):
        self.solutions = []

    @staticmethod
    def estimate_n(probability: float, threshold: float):
        return log(1 - threshold) / log(1 - probability)

    def solve(self, graph: Graph, k: Union[int, float], num_iterations=1000, num_repetitions: int = 1,
              num_attempts: int = 10):
        kn = self._adjust_k(graph, k)

        start_repetitions = perf_counter()

        nodes = list(graph.nodes())
        success_count = 0
        operations_count = 0
        iterations_checked = []
        operations_count = 0
        example_solution = None

        n = len(nodes)
        max_combinations = comb(n, kn)

        if isinstance(num_iterations, float):
            iterations_percentage = num_iterations
            num_iterations = max(int(num_iterations * max_combinations), 10)
        else:
            iterations_percentage = num_iterations / max_combinations

        if kn == 0:
            return Solution(
                graph=graph,
                k=k,
                kn=kn,
                success=True,
                success_probability=1,
                success_count=1,
                vertices=[],
                total_time=0,
                average_repetition_time=0,
                min_success_iteration=0,
                max_success_iteration=0,
                average_success_iteration=0,
                average_iterations=0,
                iterations=num_iterations,
                iterations_percentage=iterations_percentage,
                repetitions=num_repetitions
            )

        if kn > n or n == 0:
            return Solution(
                graph=graph,
                k=k,
                kn=kn,
                success=False,
                success_probability=0,
                success_count=0,
                vertices=example_solution,
                total_time=0,
                average_repetition_time=0,
                min_success_iteration=None,
                max_success_iteration=None,
                average_success_iteration=None,
                average_iterations=0,
                iterations=num_iterations,
                iterations_percentage=iterations_percentage,
                repetitions=num_repetitions
            )

        for repetition in range(num_repetitions):
            selected_vertices_set = set()
            for iteration in range(num_iterations):

                if len(selected_vertices_set) == max_combinations:
                    break

                selected_vertices = random.sample(nodes, kn)

                operations_count += n

                for _ in range(num_attempts):
                    if set(selected_vertices) not in selected_vertices_set:
                        selected_vertices_set.add(frozenset(selected_vertices))
                        break
                    selected_vertices = random.sample(nodes, kn)

                is_independent_set = True
                for i, u in enumerate(selected_vertices):
                    for v in selected_vertices[i + 1:]:
                        operations_count += 1
                        if graph.has_edge(u, v):
                            is_independent_set = False
                            break

                operations_count += 1

                if is_independent_set:
                    success_count += 1
                    iterations_checked.append(iteration + 1)
                    example_solution = selected_vertices
                    break

        end_repetitions = perf_counter()

        total_time = end_repetitions - start_repetitions

        average_repetition_time = total_time / num_repetitions if num_repetitions > 0 else 0
        success_probability = success_count / num_repetitions if num_repetitions > 0 else 0

        min_iteration_for_success = min(iterations_checked) if len(iterations_checked) > 0 else -1
        max_iteration_for_success = max(iterations_checked) if len(iterations_checked) > 0 else -1
        average_iteration_for_success = sum(iterations_checked) / len(iterations_checked) if len(
            iterations_checked) > 0 else -1
        average_iterations = sum(iterations_checked + [num_iterations]
                                 * (num_repetitions - len(
            iterations_checked))) / num_repetitions if num_repetitions > 0 else 0

        average_operations_count = operations_count / num_repetitions

        solution = Solution(
            graph=graph,
            k=k,
            kn=kn,
            success=success_probability > 0,
            success_probability=success_probability,
            success_count=success_count,
            vertices=list(example_solution) if example_solution is not None else None,
            total_time=total_time,
            average_repetition_time=average_repetition_time,
            min_success_iteration=min_iteration_for_success,
            max_success_iteration=max_iteration_for_success,
            average_success_iteration=average_iteration_for_success,
            average_iterations=average_iterations,
            iterations=num_iterations,
            iterations_percentage=iterations_percentage,
            repetitions=num_repetitions,
            average_operations_count=average_operations_count
        )

        self.solutions.append(solution)

        return solution

    def solve_at_checkpoints(self, graph: Graph, k: Union[int, float], check_points=None,
                             num_repetitions: int = 1, num_attempts: int = 10):
        if check_points is None:
            check_points = [1000, 2500, 5000, 10000]

        # check if check_points contains float values

        kn = self._adjust_k(graph, k)

        nodes = list(graph.nodes())
        success_count_at_checkpoints = [0] * len(check_points)
        iterations_checked_at_checkpoints = [[] for _ in range(len(check_points))]
        example_solution_at_checkpoints = [None] * len(check_points)
        repetition_times_at_checkpoints = [[] for _ in range(len(check_points))]
        operations_count_at_checkpoints = [[] for _ in range(len(check_points))]

        n = len(nodes)
        max_combinations = comb(n, kn)

        if any(isinstance(check_point, float) for check_point in check_points):
            iteration_percentages = check_points
            check_points = [int(check_point * max_combinations) for check_point in check_points]
        else:
            iteration_percentages = [check_point / max_combinations for check_point in check_points]

        if kn == 0:
            return Solution(
                graph=graph,
                k=k,
                kn=kn,
                success=True,
                success_probability=1,
                success_count=1,
                vertices=[],
                total_time=0,
                average_repetition_time=0,
                min_success_iteration=0,
                max_success_iteration=0,
                average_success_iteration=0,
                average_iterations=0,
                iterations=max(check_points),
                iterations_percentage=max(iteration_percentages),
                repetitions=num_repetitions,
                average_operations_count=0
            )

        if kn > n or n == 0:
            return Solution(
                graph=graph,
                k=k,
                kn=kn,
                success=False,
                success_probability=0,
                success_count=0,
                vertices=example_solution_at_checkpoints,
                total_time=0,
                average_repetition_time=0,
                min_success_iteration=None,
                max_success_iteration=None,
                average_success_iteration=None,
                average_iterations=0,
                iterations=max(check_points),
                iterations_percentage=max(iteration_percentages),
                repetitions=num_repetitions,
                average_operations_count=0
            )

        for repetition in range(num_repetitions):
            start_repetition = perf_counter()
            selected_vertices_set = set()
            operations_count = 0
            for iteration in range(max(check_points)):

                if len(selected_vertices_set) == max_combinations:
                    break

                selected_vertices = random.sample(nodes, kn)

                operations_count += n

                for _ in range(num_attempts):
                    if set(selected_vertices) not in selected_vertices_set:
                        selected_vertices_set.add(frozenset(selected_vertices))
                        break
                    selected_vertices = random.sample(nodes, kn)

                is_independent_set = True
                for i, u in enumerate(selected_vertices):
                    for v in selected_vertices[i + 1:]:
                        operations_count += 1
                        if graph.has_edge(u, v):
                            is_independent_set = False
                            break

                operations_count += 1

                if is_independent_set:
                    for i, check_point in enumerate(check_points):
                        if iteration < check_point:
                            success_count_at_checkpoints[i] += 1
                            iterations_checked_at_checkpoints[i].append(iteration + 1)
                            example_solution_at_checkpoints[i] = selected_vertices
                            operations_count_at_checkpoints[i].append(operations_count)
                            repetition_times_at_checkpoints[i].append(perf_counter() - start_repetition)
                    break

                if (iteration + 1) in check_points:
                    operations_count_at_checkpoints[check_points.index(iteration + 1)].append(operations_count)
                    repetition_times_at_checkpoints[check_points.index(iteration + 1)].append(
                        perf_counter() - start_repetition)

        total_time_at_checkpoints = [sum(repetition_times) for repetition_times in repetition_times_at_checkpoints]

        average_repetition_time_at_checkpoints = [
            sum(repetition_times) / len(repetition_times) if len(repetition_times) > 0 else 0
            for repetition_times in repetition_times_at_checkpoints]

        success_probability_at_checkpoints = [success_count / num_repetitions if num_repetitions > 0 else 0
                                              for success_count in success_count_at_checkpoints]

        min_iteration_for_success_at_checkpoints = [min(iterations_checked) if len(iterations_checked) > 0 else -1
                                                    for iterations_checked in iterations_checked_at_checkpoints]

        max_iteration_for_success_at_checkpoints = [max(iterations_checked) if len(iterations_checked) > 0 else -1
                                                    for iterations_checked in iterations_checked_at_checkpoints]

        average_iteration_for_success_at_checkpoints = [sum(iterations_checked) / len(iterations_checked) if len(
            iterations_checked) > 0 else -1
                                                        for iterations_checked in iterations_checked_at_checkpoints]

        average_iterations_at_checkpoints = [sum(iterations_checked + [check_point]
                                                 * (num_repetitions - len(
            iterations_checked))) / num_repetitions if num_repetitions > 0 else 0
                                             for check_point, iterations_checked in
                                             zip(check_points, iterations_checked_at_checkpoints)]
        average_operations_count_at_checkpoints = [sum(operations_count) / len(operations_count) if len(operations_count) > 0 else 0
                                                    for operations_count in operations_count_at_checkpoints]
        solution_at_checkpoints = [Solution(
            graph=graph,
            k=k,
            kn=kn,
            success=success_probability > 0,
            success_probability=success_probability,
            success_count=success_count,
            vertices=list(example_solution) if example_solution is not None else None,
            total_time=total_time,
            average_repetition_time=average_repetition_time,
            min_success_iteration=min_iteration_for_success,
            max_success_iteration=max_iteration_for_success,
            average_success_iteration=average_iteration_for_success,
            average_iterations=average_iterations,
            iterations=check_point,
            iterations_percentage=iteration_percentage,
            repetitions=num_repetitions,
            average_operations_count=operations_count
        ) for success_probability, success_count, example_solution, total_time, average_repetition_time,
              min_iteration_for_success, max_iteration_for_success, average_iteration_for_success, average_iterations,
              check_point, iteration_percentage, operations_count in
                zip(success_probability_at_checkpoints, success_count_at_checkpoints,
                 example_solution_at_checkpoints, total_time_at_checkpoints,
                 average_repetition_time_at_checkpoints, min_iteration_for_success_at_checkpoints,
                 max_iteration_for_success_at_checkpoints, average_iteration_for_success_at_checkpoints,
                 average_iterations_at_checkpoints, check_points, iteration_percentages, average_operations_count_at_checkpoints)]

        self.solutions.extend(solution_at_checkpoints)

        return solution_at_checkpoints

    @staticmethod
    def _adjust_k(graph: Graph, k: Union[int, float]):
        k = min(max(0, int(k * graph.number_of_nodes())), graph.number_of_nodes())
        return int(k) if isinstance(k, float) else k

    def table(self):
        print(
            f"{'Graph':<25}{'Number of Nodes':<30}{'Edge Percentage':<30}{'Number of Edges':<30}"
            f"{'k':<20}{'kn':<20}{'Iterations':<20}{'Iterations Percentage':<30}{'Repetitions':<20}"
            f"{'Total Time':<20}{'Average Repetition Time':<30}{'Success Count':<20}"
            f"{'Success Probability':<25}{'Min Success Iteration':<30}{'Max Success Iteration':<30}"
            f"{'Average Success Iteration':<30}{'Average Iterations':<30}{'Operations Count':<30}{'Success':<20}{'Vertices':<20}")

        for solution in self.solutions:
            print(solution)
            print(
                f"{solution.graph.name:<25}{solution.graph.number_of_nodes():<30}{solution.graph.edge_percentage():<30}"
                f"{solution.graph.number_of_edges():<30}{solution.k:<20}{solution.kn:<20}{solution.iterations:<20}{solution.iterations_percentage:<30}{solution.repetitions:<20}"
                f"{solution.total_time:<20.4f}{solution.average_repetition_time:<30.4f}{solution.success_count:<20}"
                f"{solution.success_probability:<25.4f}{solution.min_success_iteration:<30}{solution.max_success_iteration:<30}"
                f"{solution.average_success_iteration:<30.4f}{solution.average_iterations:<30.4f}{solution.average_operations_count:<30.4f}{solution.success:<20}{str(solution.vertices):<20}")
        print()

    def flush(self):
        self.solutions.clear()

    def save(self, path: str = 'results.csv'):
        data = {
            'Graph': [],
            'Number of Nodes': [],
            'Edge Percentage': [],
            'Number of Edges': [],
            'k': [],
            'kn': [],
            'Iterations': [],
            'Iterations Percentage': [],
            'Repetitions': [],
            'Total Time': [],
            'Average Repetition Time': [],
            'Success Count': [],
            'Success Probability': [],
            'Success': [],
            'Min Success Iteration': [],
            'Max Success Iteration': [],
            'Average Success Iteration': [],
            'Average Iterations': [],
            'Average Operations Count': [],
            'Vertices': []
        }

        for solution in self.solutions:
            data['Graph'].append(solution.graph.name)
            data['Number of Nodes'].append(solution.graph.number_of_nodes())
            data['Edge Percentage'].append(solution.graph.edge_percentage())
            data['Number of Edges'].append(solution.graph.number_of_edges())
            data['k'].append(solution.k)
            data['kn'].append(solution.kn)
            data['Iterations'].append(solution.iterations)
            data['Iterations Percentage'].append(solution.iterations_percentage)
            data['Repetitions'].append(solution.repetitions)
            data['Total Time'].append(solution.total_time)
            data['Average Repetition Time'].append(solution.average_repetition_time)
            data['Success Count'].append(solution.success_count)
            data['Success Probability'].append(solution.success_probability)
            data['Success'].append(solution.success)
            data['Min Success Iteration'].append(solution.min_success_iteration)
            data['Max Success Iteration'].append(solution.max_success_iteration)
            data['Average Success Iteration'].append(solution.average_success_iteration)
            data['Average Iterations'].append(solution.average_iterations)
            data['Average Operations Count'].append(solution.average_operations_count)
            data['Vertices'].append(str(solution.vertices))

        df = pd.DataFrame(data)
        df.replace({None: np.nan, 'None': np.nan, -1: np.nan, '-1': np.nan}, inplace=True)
        df.to_csv(path, index=False)


class GraphMapping:

    def __init__(self, graphs: List[Graph]):
        self.graphs = graphs

    def map(self):
        for graph in self.graphs:
            print(f"{graph.name} -> {graph.path}")

    def save(self, path: str = 'mapping.csv', append: bool = False):
        data = {
            'Graph': [],
            'Path': [],
        }

        for graph in self.graphs:
            data['Graph'].append(graph.name)
            data['Path'].append(graph.path)

        df = pd.DataFrame(data)

        if append and os.path.isfile(path):
            df.to_csv(path, index=False, mode='a', header=False)
        else:
            df.to_csv(path, index=False)


def load_sw_graphs():
    SW_FILES = [
        'collections/SW_ALGUNS_GRAFOS/SWtinyG.txt',
        'collections/SW_ALGUNS_GRAFOS/SWmediumG.txt',
    ]
    graphs = Graph.from_files(SW_FILES, mode='dwneV')
    return graphs


def load_git_graphs():
    GIT_FILES = ['collections/git_web_ml/musae_git_edges.txt']
    graphs = Graph.from_files(GIT_FILES, mode='dwneV')
    return graphs


def load_twitch_graphs():
    TWITCH_FILES = ['collections/twitch_gamers/large_twitch_edges.txt']
    graphs = Graph.from_files(TWITCH_FILES, mode='dwneV')
    return graphs


def load_danie_graphs(grange):
    start, end = grange
    DANIE_FILES = glob.glob("collections/graphs/*.gml")
    start = 0 if start is None else start * 4 - 16
    end = len(DANIE_FILES) if end is None else end * 4 - 16
    graphs = sorted(DANIE_FILES, key=sort_by_natural_key)[start:end]
    graphs = Graph.from_files(graphs)
    return graphs


def load_cd_graphs(group=None):
    CD_FILES = glob.glob("collections/random_graphs/BD*/Grafo*.txt") \
        if group is None else glob.glob(f"collections/random_graphs/BD{group}/Grafo*.txt")
    graphs = Graph.from_files(sorted(CD_FILES, key=sort_by_natural_key), mode='neA')
    return graphs


def load_custom_graphs(file, mode='dwneV'):
    graphs = Graph.from_files([file], mode=mode)
    return graphs


def load_n_danie_graphs(n):
    DANIE_FILES = glob.glob("collections/graphs/*.gml")
    graphs = Graph.from_files(random.sample(DANIE_FILES, n))
    return graphs


def sort_by_natural_key(s):
    # Define a regular expression to match a number (integer or float)
    pattern = re.compile(r'(\d+\.\d+|\d+)')
    # Use the regular expression to split the string
    parts = pattern.split(s)
    # Convert numeric parts to float, leave non-numeric parts as strings
    parts = [float(part) if part.replace('.', '').isdigit() else part for part in parts]
    return parts


def load_graphs(space, group=None, mode='dwneV', grange=(None, None)):
    match space:
        case 'sw':
            return load_sw_graphs()
        case 'git':
            return load_git_graphs()
        case 'twitch':
            return load_twitch_graphs()
        case 'danie':
            return load_danie_graphs(grange)
        case 'cd':
            return load_cd_graphs(group)
        case _:
            return load_custom_graphs(space, mode)


def main():
    parser = argparse.ArgumentParser(description='Graph Solver with argparse')
    parser.add_argument('--graphs', nargs='+', choices=['danie', 'sw', 'git', 'twitch', 'cd'],
                        default=['danie'],
                        help='Select the graphs to load. Options: danie, sw, git, twitch, cd. Default: danie')
    parser.add_argument('--file', type=str, default=None,
                        help='Select the file to load. Default: None')
    parser.add_argument('--mode', type=str, default='dwneV',
                        help='Select the mode to load the graphs. Options: V, dwneV, neA. Default: dwneV')
    parser.add_argument('--group', type=str, default=None,
                        help='Select the group of graphs to load. Only works with cd. Options: 1, 2, 3, 4, 5, 6. Default: None')
    parser.add_argument('--repetitions', help='List of repetitions to run', nargs='+', type=int, default=[10])
    parser.add_argument('--iterations', help='List of iterations to run', nargs='+', type=float,
                        default=[50, 100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
    parser.add_argument('--k', help='List of k values to run', nargs='+', type=float, default=[0.125, 0.25, 0.5, 0.75])
    parser.add_argument('--name', type=str, default=None,
                        help='Name of the output files. Default: None')
    parser.add_argument('--from', type=int, default=None,
                        help='Start index to load the graphs. Default: None')
    parser.add_argument('--to', type=int, default=None,
                        help='End index to load the graphs. Default: None')

    args = parser.parse_args()

    graph_range = (args.__dict__['from'], args.__dict__['to'])
    graph_files = args.graphs[0] if not args.file else args.file
    suffix = f"_{args.name}" if args.name else ""

    graphs = load_graphs(graph_files, args.group, args.mode, graph_range)
    # graphs = load_n_danie_graphs(3)
    # graphs = load_graphs('danie', mode='dwneV', grange=(None, 14))

    mapping = GraphMapping(graphs)
    mapping.map()
    mapping.save(f'mapping{suffix}.csv')

    solver = ISDPSolver()

    atexit.register(solver.table)
    atexit.register(lambda: solver.save(f"results{suffix}.csv"))

    k_values = args.k
    num_iterations_values = args.iterations
    num_repetitions_values = args.repetitions

    for graph in graphs:
        logging.info(f"Solving {graph.name} -> {graph.path}")
        for k in k_values:
            solutions = solver.solve_at_checkpoints(graph, k, num_repetitions=max(num_repetitions_values),
                                                    check_points=num_iterations_values)
            for solution in solutions:
                logging.info(
                    f"Found solution for num_iterations={solution.iterations}: {solution} in {solution.total_time:.4f} seconds.")

            # for num_iterations in num_iterations_values:
            #     for num_repetitions in num_repetitions_values:
            #         logging.info(f"Trying k={k}, num_iterations={num_iterations}, num_repetitions={num_repetitions}...")
            #         solution = solver.solve(graph, k, num_iterations=num_iterations, num_repetitions=num_repetitions)
            #         logging.info(f"Found solution: {solution} in {solution.total_time:.4f} seconds.")
            #         # solution.visualize()

    # remember to fix average_repetition_time to value * num_iterations / num_repetitions


if __name__ == '__main__':
    main()
