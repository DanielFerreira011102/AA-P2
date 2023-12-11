from ast import literal_eval

import pandas as pd
from analyzer import InteractiveAnalyzer

df_brute_force = pd.read_csv('pout/results_brute_force.csv', sep=';')
df_clever = pd.read_csv('pout/results_clever.csv', sep=';')
df_greedy_v1 = pd.read_csv('pout/results_greedy_v1.csv', sep=';')
df_greedy_v2 = pd.read_csv('pout/results_greedy_v2.csv', sep=';')

df_random_danie_fixed = pd.read_csv('out/danie/results_danie_fixed.csv')
df_random_danie_percentage = pd.read_csv('out/danie/results_danie_percentage.csv')
df_random_cd0 = pd.read_csv('out/cd0/results_cd0.csv')
df_random_cd1 = pd.read_csv('out/cd1/results_cd1.csv')
df_random_sw = pd.read_csv('out/sw/results_sw.csv')


def main():
    data = {
        'brute_force': df_brute_force,
        'branching': df_clever,
        'greedy_v1': df_greedy_v1,
        'greedy_v2': df_greedy_v2,
        'random_fixed': df_random_danie_fixed,
        'random_percentage': df_random_danie_percentage,
    }

    def convert_to_list_or_none(value):
        if pd.isna(value):
            return None
        return literal_eval(value)

    options = {
        'metadata': {
            'brute_force': {
                'name': 'Brute Force',
            },
            'branching': {
                'name': 'Branching',
            },
            'greedy_v1': {
                'name': 'Greedy',
            },
            'greedy_v2': {
                'name': 'Improved Greedy',
            },
            'random_fixed': {
                'name': 'Randomized ({iterations})',
            },
            'random_percentage': {
                'name': 'Randomized ({iterations}%)',
            },
        },
        'rename': {
            'random_fixed': {
                'result': 'Success',
                'nodes': 'Number of Nodes',
                'edge_percentage': 'Edge Percentage',
                'elapsed_time': 'Average Repetition Time',
                'operations_count': 'Average Operations Count',
            },
            'random_percentage': {
                'result': 'Success',
                'nodes': 'Number of Nodes',
                'edge_percentage': 'Edge Percentage',
                'elapsed_time': 'Average Repetition Time',
                'operations_count': 'Average Operations Count',
            },
        },
        'apply': {
            'random_fixed': [
                {'column': 'Vertices', 'function': convert_to_list_or_none},
            ],
            'random_percentage': [
                {'column': 'Vertices', 'function': convert_to_list_or_none},
            ],
        },
        'replace': {
            'brute_force': [
                {'column': 'result', 'values': {0: False, 1: True}},
            ],
            'branching': [
                {'column': 'result', 'values': {0: False, 1: True}},
            ],
            'greedy_v1': [
                {'column': 'result', 'values': {0: False, 1: True}},
            ],
            'greedy_v2': [
                {'column': 'result', 'values': {0: False, 1: True}},
            ],
        },
    }

    analyzer = InteractiveAnalyzer(data, options)

    analyzer.run()


if __name__ == '__main__':
    main()
