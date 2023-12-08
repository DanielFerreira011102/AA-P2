import itertools
import math
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import default_rng

df1 = pd.read_csv('out/danie/results_danie_fixed.csv')
df2 = pd.read_csv('results_danie_p1.csv')

# add the 'Average Operations Count' column in df2 to df1 to the second to last column
df1.insert(len(df1.columns) - 1, 'Average Operations Count', df2['Average Operations Count'])

# save the result
df1.to_csv('results_danie_p2.csv', index=False)
