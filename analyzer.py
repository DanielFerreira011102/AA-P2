import inspect
import os
import re
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.special import comb
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')


class InteractiveAnalyzer:

    def __init__(self, dataframes, options):
        self.dataframes = dataframes
        self.options = options
        self.rename_columns()
        self.apply_functions()
        self.replace_values()

    @staticmethod
    def accepted_columns():
        return ['elapsed_time', 'k', 'nodes', 'edge_percentage', 'result']

    def apply_functions(self):
        for df_name, apply_list in self.options.get('apply', {}).items():
            if df_name in self.dataframes:
                df = self.dataframes[df_name]
                for apply_dict in apply_list:
                    column = apply_dict.get('column', None)
                    function = apply_dict.get('function', None)
                    if column is not None and function is not None and column in df.columns:
                        print(f"Applying function {function.__name__} to column {column} in dataframe {df_name}")
                        df[column] = df[column].apply(function)

    def replace_values(self):
        for df_name, replace_list in self.options.get('replace', {}).items():
            if df_name in self.dataframes:
                df = self.dataframes[df_name]
                for replace_dict in replace_list:
                    column = replace_dict.get('column', None)
                    values = replace_dict.get('values', None)
                    if column is not None and values is not None and column in df.columns:
                        print(f"Replacing values in column {column} in dataframe {df_name}")
                        df[column] = df[column].replace(values)

    def rename_columns(self):
        for df_name, column_mapping in self.options.get('columns', {}).items():
            if df_name in self.dataframes:
                df = self.dataframes[df_name]
                for column, assigned_column in column_mapping.items():
                    if assigned_column in df.columns:
                        print(f"Renaming column {assigned_column} to {column} in dataframe {df_name}")
                        df.rename(columns={assigned_column: column}, inplace=True)

    def stats_table(self, title):
        df = self.dataframes.get(title, None)

        if df is None:
            print(f"Invalid dataframe: {title}")
            return

        df = self.select_iterations(df, title)

        unique_k_values = sorted(df['k'].unique())
        unique_edge_percentage_values = sorted(df['edge_percentage'].unique())

        print("+" + "-" * 92 + "+")
        print("| {:^90} |".format(title))
        print("+" + "-" * 92 + "+")
        print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("label", "value", "min", "max", "median",
                                                                            "mean", "std"))
        print("+" + "-" * 92 + "+")

        for k in unique_k_values:
            sub_df = df[df['k'] == k]

            elapsed_time_column = sub_df['elapsed_time']

            min_val = elapsed_time_column.min()
            max_val = elapsed_time_column.max()
            median_val = elapsed_time_column.median()
            mean_val = elapsed_time_column.mean()
            std_val = elapsed_time_column.std()

            if k == unique_k_values[len(unique_k_values) // 2]:
                print("| {:<12} {:<12} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} |".format(
                    "k", k, min_val, max_val, median_val, mean_val, std_val))
            else:
                print(
                    "| {:<12} {:<12} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} |".format(
                        "", k, min_val, max_val, median_val, mean_val, std_val))

        print("+" + "-" * 92 + "+")
        for edge_percentage in unique_edge_percentage_values:
            sub_df = df[df['edge_percentage'] == edge_percentage]

            elapsed_time_column = sub_df['elapsed_time']

            min_val = elapsed_time_column.min()
            max_val = elapsed_time_column.max()
            median_val = elapsed_time_column.median()
            mean_val = elapsed_time_column.mean()
            std_val = elapsed_time_column.std()

            if edge_percentage == unique_edge_percentage_values[len(unique_edge_percentage_values) // 2]:
                print("| {:<12} {:<12} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} |".format(
                    "ep", edge_percentage, min_val, max_val, median_val, mean_val, std_val))
            else:
                print(
                    "| {:<12} {:<12} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} |".format(
                        "", edge_percentage, min_val, max_val, median_val, mean_val, std_val))

        print("+" + "-" * 92 + "+")
        print()

    def confusion(self, df_model_title, df_to_compare_title):
        df_model = self.dataframes.get(df_model_title, None)
        df_to_compare = self.dataframes.get(df_to_compare_title, None)

        if df_model is None or df_to_compare is None:
            print(f"Invalid dataframe: {df_model_title}")
            return

        df_model = self.select_iterations(df_model, df_model_title)
        df_to_compare = self.select_iterations(df_to_compare, df_to_compare_title)

        print(df_model)
        print(df_to_compare)
        print(type(df_model))
        print(type(df_to_compare))

        merged_df = pd.merge(df_model, df_to_compare, on=['edge_percentage', 'k', 'nodes'], how='inner',
                             suffixes=('_model', '_compare'))

        cm = confusion_matrix(merged_df['result_model'], merged_df['result_compare'])

        TN, FP, FN, TP = cm.ravel()

        print("+" + "-" * 79 + "+")

        print("| {:^77} |".format(df_to_compare_title))

        print("+" + "-" * 79 + "+")

        print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("label", "value", "TN", "FP", "FN", "TP"))

        print("+" + "-" * 79 + "+")

        print("| {:25} {:<12} {:<12} {:<12} {:<12} |".format("Total", TN, FP, FN, TP))

        unique_k_values = sorted(merged_df['k'].unique())
        unique_edge_percentage_values = sorted(merged_df['edge_percentage'].unique())
        unique_nodes_values = sorted(merged_df['nodes'].unique())

        print("+" + "-" * 79 + "+")

        K_CM = []

        for k in sorted(unique_k_values):
            subset_df = merged_df[merged_df['k'] == k]
            subset_sm = confusion_matrix(subset_df['result_model'], subset_df['result_compare'])
            subset_TN, subset_FP, subset_FN, subset_TP = subset_sm.ravel()

            K_CM.append([k, subset_TN, subset_FP, subset_FN, subset_TP])

            if k == unique_k_values[len(unique_k_values) // 2]:
                print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("k", k, subset_TN, subset_FP, subset_FN,
                                                                             subset_TP))
            else:
                print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("", k, subset_TN, subset_FP, subset_FN,
                                                                             subset_TP))

        print("+" + "-" * 79 + "+")

        EP_CM = []

        for edge_percentage in sorted(unique_edge_percentage_values):
            subset_df = merged_df[merged_df['edge_percentage'] == edge_percentage]
            subset_sm = confusion_matrix(subset_df['result_model'], subset_df['result_compare'])
            subset_TN, subset_FP, subset_FN, subset_TP = subset_sm.ravel()

            EP_CM.append([edge_percentage, subset_TN, subset_FP, subset_FN, subset_TP])

            if edge_percentage == unique_edge_percentage_values[len(unique_edge_percentage_values) // 2]:
                print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("ep", edge_percentage, subset_TN,
                                                                             subset_FP, subset_FN, subset_TP))
            else:
                print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("", edge_percentage, subset_TN, subset_FP,
                                                                             subset_FN, subset_TP))

        print("+" + "-" * 79 + "+")

        N_CM = []

        for nodes in sorted(unique_nodes_values):
            subset_df = merged_df[merged_df['nodes'] == nodes]
            subset_sm = confusion_matrix(subset_df['result_model'], subset_df['result_compare'])
            subset_TN, subset_FP, subset_FN, subset_TP = subset_sm.ravel()

            N_CM.append([nodes, subset_TN, subset_FP, subset_FN, subset_TP])

        p_N_CM = N_CM if len(N_CM) < 6 else N_CM[:3] + [['...'] * 5] + N_CM[-3:]

        for nodes, subset_TN, subset_FP, subset_FN, subset_TP in p_N_CM:
            if nodes == p_N_CM[len(p_N_CM) // 2][0]:
                print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("n", nodes, subset_TN, subset_FP,
                                                                             subset_FN, subset_TP))
            else:
                print("| {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("", nodes, subset_TN, subset_FP,
                                                                             subset_FN, subset_TP))

        print("+" + "-" * 79 + "+")

        print()

        def find_extreme_values_and_elements(CM, key_index):
            max_value = max(CM, key=lambda x: x[key_index])[key_index]
            min_value = min(CM, key=lambda x: x[key_index])[key_index]

            max_elements = [element[0] for element in CM if element[key_index] == max_value]
            min_elements = [element[0] for element in CM if element[key_index] == min_value]

            return max_value, max_elements, min_value, min_elements

        MAX_TP_K, ELEMENTS_MAX_TP_K, MIN_TP_K, ELEMENTS_MIN_TP_K = find_extreme_values_and_elements(K_CM, 4)
        MAX_TP_EP, ELEMENTS_MAX_TP_EP, MIN_TP_EP, ELEMENTS_MIN_TP_EP = find_extreme_values_and_elements(EP_CM, 4)
        MAX_TP_N, ELEMENTS_MAX_TP_N, MIN_TP_N, ELEMENTS_MIN_TP_N = find_extreme_values_and_elements(N_CM, 4)

        MAX_FP_K, ELEMENTS_MAX_FP_K, MIN_FP_K, ELEMENTS_MIN_FP_K = find_extreme_values_and_elements(K_CM, 2)
        MAX_FP_EP, ELEMENTS_MAX_FP_EP, MIN_FP_EP, ELEMENTS_MIN_FP_EP = find_extreme_values_and_elements(EP_CM, 2)
        MAX_FP_N, ELEMENTS_MAX_FP_N, MIN_FP_N, ELEMENTS_MIN_FP_N = find_extreme_values_and_elements(N_CM, 2)

        MAX_FN_K, ELEMENTS_MAX_FN_K, MIN_FN_K, ELEMENTS_MIN_FN_K = find_extreme_values_and_elements(K_CM, 3)
        MAX_FN_EP, ELEMENTS_MAX_FN_EP, MIN_FN_EP, ELEMENTS_MIN_FN_EP = find_extreme_values_and_elements(EP_CM, 3)
        MAX_FN_N, ELEMENTS_MAX_FN_N, MIN_FN_N, ELEMENTS_MIN_FN_N = find_extreme_values_and_elements(N_CM, 3)

        MAX_TN_K, ELEMENTS_MAX_TN_K, MIN_TN_K, ELEMENTS_MIN_TN_K = find_extreme_values_and_elements(K_CM, 1)
        MAX_TN_EP, ELEMENTS_MAX_TN_EP, MIN_TN_EP, ELEMENTS_MIN_TN_EP = find_extreme_values_and_elements(EP_CM, 1)
        MAX_TN_N, ELEMENTS_MAX_TN_N, MIN_TN_N, ELEMENTS_MIN_TN_N = find_extreme_values_and_elements(N_CM, 1)

        def print_tf_table(category, labels, max_values, elements_max, min_values, elements_min):
            for label, max_value, elements_max, min_value, elements_min in zip(labels, max_values, elements_max,
                                                                               min_values, elements_min):

                elements_max_display = elements_max[:3] + ["..."] + elements_max[-3:] if len(
                    elements_max) > 6 else elements_max

                elements_min_display = elements_min[:3] + ["..."] + elements_min[-3:] if len(
                    elements_min) > 6 else elements_min

                if label == labels[len(labels) // 2]:
                    print("| {:<12} {:<12} {:<12} {:<32} {:<12} {:<32} |".format(category, label, max_value,
                                                                                 ','.join(str(element) for element in
                                                                                          elements_max_display),
                                                                                 min_value, ','.join(
                            str(element) for element in elements_min_display)))
                else:
                    print("| {:<12} {:<12} {:<12} {:<32} {:<12} {:<32} |".format("", label, max_value,
                                                                                 ','.join(str(element) for element in
                                                                                          elements_max_display),
                                                                                 min_value, ','.join(
                            str(element) for element in elements_min_display)))

            print("+" + "-" * 119 + "+")

        print("+" + "-" * 119 + "+")
        print("| {:^117} |".format(df_to_compare_title))
        print("+" + "-" * 119 + "+")
        print(
            "| {:12} {:<12} {:<12} {:<32} {:<12} {:<32} |".format("term", "category", "max", "max elements", "min",
                                                                  "min elements"))
        print("+" + "-" * 119 + "+")

        print_tf_table("TP", ['k', 'ep', 'n'], [MAX_TP_K, MAX_TP_EP, MAX_TP_N],
                       [ELEMENTS_MAX_TP_K, ELEMENTS_MAX_TP_EP, ELEMENTS_MAX_TP_N],
                       [MIN_TP_K, MIN_TP_EP, MIN_TP_N], [ELEMENTS_MIN_TP_K, ELEMENTS_MIN_TP_EP, ELEMENTS_MIN_TP_N])
        print_tf_table("FP", ['k', 'ep', 'n'], [MAX_FP_K, MAX_FP_EP, MAX_FP_N],
                       [ELEMENTS_MAX_FP_K, ELEMENTS_MAX_FP_EP, ELEMENTS_MAX_FP_N],
                       [MIN_FP_K, MIN_FP_EP, MIN_FP_N], [ELEMENTS_MIN_FP_K, ELEMENTS_MIN_FP_EP, ELEMENTS_MIN_FP_N])
        print_tf_table("FN", ['k', 'ep', 'n'], [MAX_FN_K, MAX_FN_EP, MAX_FN_N],
                       [ELEMENTS_MAX_FN_K, ELEMENTS_MAX_FN_EP, ELEMENTS_MAX_FN_N],
                       [MIN_FN_K, MIN_FN_EP, MIN_FN_N], [ELEMENTS_MIN_FN_K, ELEMENTS_MIN_FN_EP, ELEMENTS_MIN_FN_N])
        print_tf_table("TN", ['k', 'ep', 'n'], [MAX_TN_K, MAX_TN_EP, MAX_TN_N],
                       [ELEMENTS_MAX_TN_K, ELEMENTS_MAX_TN_EP, ELEMENTS_MAX_TN_N],
                       [MIN_TN_K, MIN_TN_EP, MIN_TN_N], [ELEMENTS_MIN_TN_K, ELEMENTS_MIN_TN_EP, ELEMENTS_MIN_TN_N])
        print()

        # table for accuracy, precision, recall, f1-score, tp rate (sensitivity), fp rate, tn rate (specificity), fn rate, youden's index
        print("+" + "-" * 144 + "+")
        print("| {:^142} |".format(df_to_compare_title))
        print("+" + "-" * 144 + "+")
        print(
            "| {:12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} |".format("term", "category",
                                                                                                     "accuracy",
                                                                                                     "precision",
                                                                                                     "recall",
                                                                                                     "f1-score",
                                                                                                     "tp rate",
                                                                                                     "fp rate",
                                                                                                     "tn rate",
                                                                                                     "fn rate",
                                                                                                     "J"))
        print("+" + "-" * 144 + "+")

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        tp_rate = recall
        fp_rate = FP / (FP + TN)
        tn_rate = TN / (FP + TN)
        fn_rate = FN / (TP + FN)
        youdens_index = tp_rate + tn_rate - 1

        print(
            "| {:<12} {:<12} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} |".format(
                "Total", "", accuracy, precision, recall, f1_score, tp_rate, fp_rate, tn_rate, fn_rate, youdens_index))

        print("+" + "-" * 144 + "+")

        print()

    def visualize_elapsed_time(self, title, path, log, window_size=1):
        df = self.dataframes.get(title, None)

        df, iterations = self.select_iterations(df, title, return_iterations=True)

        unique_k_values = df['k'].unique()

        for k in unique_k_values:
            fig, ax = plt.subplots(figsize=(6, 5))

            sorted_edge_percentages = sorted(df['edge_percentage'].unique())
            colors = sns.color_palette("tab10", n_colors=len(sorted_edge_percentages) + 1)

            for edge_percentage, color in zip(sorted_edge_percentages, colors):
                sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
                smoothed_elapsed_time = sub_df['elapsed_time'].rolling(window=window_size).mean()
                smoothed_elapsed_time = np.log2(smoothed_elapsed_time + 0.01) if log else smoothed_elapsed_time
                ax.plot(sub_df['nodes'], smoothed_elapsed_time,
                        label=f'$\\varepsilon_p = {edge_percentage}$', marker='o', color=color)

            ax.set_xlabel('n')
            ax.set_ylabel('log2(elapsed time (s))' if log else 'elapsed time (s)')
            ax.legend(loc='upper left')
            ax.grid()
            plt.tight_layout()

            if path is not None:
                plt.savefig(os.path.join(path, f'{title}_{k}_etvsnnddkvaep' + ('_log' if log else '') + '.png'),
                            dpi=300, bbox_inches='tight')
            plt.show()

    def visualize_forecast_elapsed_time(self, title, path, log, p0=None, maxfev=100000):
        df = self.dataframes.get(title, None)

        df, iterations = self.select_iterations(df, title, return_iterations=True)

        unique_k_values = df['k'].unique()

        def formal(n, a, b, k):
            return a * n * iterations * k ** 2 + b

        arg_count = len(inspect.signature(formal).parameters)

        p0 = [1] * (arg_count - 1) if p0 is None else p0

        p0.pop(0)

        for k in unique_k_values:
            fig, ax = plt.subplots(figsize=(6, 5))

            sorted_edge_percentages = sorted(df['edge_percentage'].unique())
            colors = sns.color_palette("tab10", n_colors=len(sorted_edge_percentages) + 1)

            formal = partial(formal, k=k)

            for edge_percentage, color in zip(sorted_edge_percentages, colors):
                sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
                x_values = sub_df['nodes']
                y_values = sub_df['elapsed_time']
                popt = curve_fit(formal, x_values, y_values, p0=p0, maxfev=maxfev)

                forecast_x_values = np.array([i for i in range(4, 1000)])
                forecast_y_values = np.array([formal(i, *popt[0]) for i in forecast_x_values])
                r2 = 1 - np.sum((y_values - formal(x_values, *popt[0])) ** 2) / np.sum(
                    (y_values - np.mean(y_values)) ** 2)

                print("For k = ", k, " and edge percentage = ", edge_percentage)
                print("a = ", popt[0][0])
                print("b = ", popt[0][1])
                print("R2 = ", r2)

                forecast_y_values = np.log2(forecast_y_values + 0.01) if log else forecast_y_values
                ax.plot(forecast_x_values, forecast_y_values,
                        label=f'$\\varepsilon_p = {edge_percentage}$', color=color)

            ax.set_xlabel('n')
            ax.set_ylabel('log2(elapsed time (s))' if log else 'elapsed time (s)')
            ax.legend(loc='upper left')
            ax.grid()
            plt.tight_layout()

            if path is not None:
                plt.savefig(os.path.join(path, f'{title}_{k}_forecast' + ('_log' if log else '') + '.png'),
                            dpi=300, bbox_inches='tight')
            plt.show()

    def run(self):
        OPTIONS = [
            (1, "Table query+", "Show stats table", r"(?:TABLE|TBL|1)\s+(.*)\s*$"),
            (2, "Confusion model query+", "Show confusion matrix", r"(?:CONFUSION|CM|2)\s+(.*)\s*$"),
            (3, "Time query+", "Show elapsed time graphs", r"(?:TIME|TM|3)\s+(.*)\s*$"),
            (4, "Forecast query+", "Show forecast elapsed time graphs", r"(?:FORECAST|FC|4)\s+(.*)\s*$"),
            (5, "Help", "Show help menu", r"(?:HELP|H|5)\s*$"),
            (6, "Quit", "Terminate the program", r"(?:QUIT|EXIT|Q|6)\s*$"),
            (7, "Clear", "Clear the screen", r"(?:CLEAR|CLS|7)\s*$")
        ]

        def _render_options():
            opt_length = max(len(opt) for _, opt, _, _ in OPTIONS) + 16
            option_templates = [
                f"{oid}. {opt.ljust(opt_length)} - {description}" for i, (oid, opt, description, _) in
                enumerate(OPTIONS)
            ]

            line_length = max(len(template) for template in option_templates)
            line_length += (line_length % 2 != 0)

            side_length = (line_length - 6) // 2
            HELP = "\n".join(option_templates)

            MENU_HEADER = f"{'+' + '-' * side_length} MENU {'-' * side_length}+"
            MENU_FOOTER = f"{'+' + '-' * line_length}+"

            return f"{MENU_HEADER}\n{HELP}\n{MENU_FOOTER}"

        def find_option(option_input):
            for oid, _, _, regex in OPTIONS:
                match = re.match(regex, option_input, re.IGNORECASE)
                if match:
                    return oid, match
            return None, None

        HELP_MENU = _render_options()

        save_files = input("Do you want to save session files? (Y/N): ")
        save_files = save_files.strip().lower() in ['y', 'yes']

        out = None
        if save_files:
            out = input("Enter the path to save the files: ")
            out = out.strip()
            os.makedirs(out, exist_ok=True)

        print()
        print(HELP_MENU)

        while True:
            option = input(": ")

            option_id, option_match = find_option(option)

            if option_id is None:
                print("Invalid input. Type 'H' or 'HELP' for menu.")
                continue

            if option_id == 1:
                dataframes = option_match.group(1).split()
                for dataframe in dataframes:
                    self.stats_table(dataframe)

            elif option_id == 2:
                dataframes = option_match.group(1).split()
                for dataframe in dataframes[1:]:
                    self.confusion(dataframes[0], dataframe)

            elif option_id == 3:
                dataframes = option_match.group(1).split()
                log_question = input("Do you want to log the graphs? (Y/N): ").strip()
                log = log_question.strip().lower() in ['y', 'yes']

                for dataframe in dataframes:
                    self.visualize_elapsed_time(dataframe, out, log)

            elif option_id == 4:
                dataframes = option_match.group(1).split()
                log_question = input("Do you want to log the graphs? (Y/N): ").strip()
                log = log_question.strip().lower() in ['y', 'yes']

                for dataframe in dataframes:
                    self.visualize_forecast_elapsed_time(dataframe, out, log)

            elif option == 5:
                print(HELP_MENU)

            elif option_id == 6:
                print("Exiting the program...")
                break

            elif option_id == 7:
                os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def select_iterations(df, title, return_iterations=False):
        iterations = None
        if title == 'random_fixed' or title == 'random_percentage':
            iterations = df['Iterations'].unique()
            iterations_percentage = df['Iterations Percentage'].unique()
            possible_iterations, iterations_type = (iterations, 'Iterations') if len(iterations) < len(
                iterations_percentage) else (iterations_percentage, 'Iterations Percentage')
            selected_iterations = eval(input(
                f"Select the number of iterations ({iterations_type}) to be used (possible values: {', '.join(str(iteration) for iteration in possible_iterations)}): "))
            if not isinstance(selected_iterations, (int, float)) or selected_iterations not in possible_iterations:
                print("Invalid input. Using the first value.")
                selected_iterations = possible_iterations[0]
            df = df[df[iterations_type] == selected_iterations]
        return df if not return_iterations else (df, iterations)
