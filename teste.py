import itertools
import math
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import default_rng


def convert_git_to_new_format():
    df = pd.read_csv('collections/git_web_ml/musae_git_edges.csv')
    # file has id_1, id_2
    # I want .txt with
    # first line 0
    # second line 0
    # third line number of nodes = 37700
    # fourth line number of edges = 289003
    # next lines are edges: vertex1 vertex2

    with open('collections/git_web_ml/musae_git_edges.txt', 'w') as f:
        f.write('0\n')
        f.write('0\n')
        f.write(f'{37700}\n')
        f.write(f'{289003}\n')
        for i in range(len(df)):
            f.write(f'{df.iloc[i, 0]} {df.iloc[i, 1]}\n')


def convert_twitch_gamers_to_new_format():
    df = pd.read_csv('collections/twitch_gamers/large_twitch_edges.csv')
    # file has id_1, id_2
    # I want .txt with
    # first line 0
    # second line 0
    # third line number of nodes = 71269
    # fourth line number of edges = 264332
    # next lines are edges: vertex1 vertex2

    with open('collections/twitch_gamers/large_twitch_edges.txt', 'w') as f:
        f.write('0\n')
        f.write('0\n')
        f.write(f'{168114}\n')
        f.write(f'{6797557}\n')
        for i in range(len(df)):
            f.write(f'{df.iloc[i, 0]} {df.iloc[i, 1]}\n')

a = 743
b = 457
c = 983
m = 9999991  # A large prime number


def run(n):
    m1 = math.comb(108, int(108 * 0.125))
    print(f"Combinations = {m1}")
    c1 = 1000000000000
    print(f"True solutions = {c1}")
    print(f"Percentage of True Solutions = {c1 / m1 * 100}%")
    for i in range(n):
        guess = random.randint(1, m1)
        if guess < c1:
            print("Guessed it at", i)
            return True
    return False

def run_n(n, k):
    suc = 0
    for i in range(n):
        suc += int(run(k))
    return suc

print(math.comb(26, 13))
# s = run_n(10, 10000)
# print(s)


rng = default_rng()
g = nx.gnp_random_graph(26, 0.75)
nodes = list(g.nodes())
n = len(nodes)

s = time.time()

for j in itertools.combinations(nodes, 13):
    for u, v in itertools.combinations(j, 2):
        if g.has_edge(nodes[u], nodes[v]):
            break

print(time.time() - s)

s = time.time()

for repetition in range(1):
    selected_vertices_set = set()
    r = list(range(26))
    for iteration in range(int(math.comb(26, 13) * 0.1)):

        if len(selected_vertices_set) == math.comb(26, 13):
            break

        is_independent_set = not any(
            g.has_edge(nodes[u], nodes[v]) for i, u in enumerate(selected_vertices) for v in
            selected_vertices[i + 1:])

        if is_independent_set:
            break

print(time.time() - s)