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

def g(nodes, n, k):
    random_combinations = []

    # Fill the reservoir with the first n combinations
    for i, combination in enumerate(itertools.combinations(nodes, k)):
        if i < n:
            random_combinations.append(combination)
        else:
            # Randomly replace elements in the reservoir with decreasing probability
            j = random.randint(0, i)
            if j < n:
                random_combinations[j] = combination

    return random_combinations

def f(nodes, n, k):
    ls = []
    for i in range(n):
        ls.append(random.sample(nodes, k))
    return ls


nn = 26
k = 13
nodes = list(range(nn))
n = int(math.comb(nn, k) * 0.1)

for i in range(3):
    s = time.perf_counter()
    random_combinations = g(nodes, n, k)
    print(random_combinations[0:5])
    print(time.perf_counter() - s)

print('-------------------')

for i in range(3):
    s = time.perf_counter()
    random_combinations = f(nodes, n, k)
    print(random_combinations[0:5])
    print(time.perf_counter() - s)