import pandas as pd
import types
import numpy as np
from numba import njit
import math
import matplotlib.pyplot as plt
from math import sin, cos, pi, e, exp
from scipy.linalg import eigh

def two():
    return 2

def ReadFile(FilePath)-> pd.DataFrame:
    return pd.read_csv(FilePath,sep = r"\s+", header=None)

def pos_real(grid: types.Iterable, k: int):
    pos = np.zeros(2)
    for i in range(4):
        pos[0] += grid.iloc[k-1, 0]