import pandas as pd
import typing
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

def pos_real(grid: pd.DataFrame, k: int):
    pos = np.zeros(2)
    for i in range(4):
        pos[0] += grid.iloc[k-1, 0]


# funkcje biliniowe
@njit
def f1(ksi):
    return (1-ksi)/2
@njit
def f2(ksi):
    return (1+ksi)/2
@njit
def gi(i:int,ksi1,ksi2):
    assert i >0 and i <5
    match i:
        case 1:
            return f1(ksi1)*f1(ksi2)
        case 2:
            return f2(ksi1)*f1(ksi2)
        case 3:
            return f1(ksi1)*f2(ksi2)
        case 4:
            return f2(ksi1)*f2(ksi2)
@njit 
def xy_nlg(k:int, i:int, nlg:np.ndarray, wezly:np.ndarray):
    k_is = nlg[:,1][nlg[:,0] == k]
    k_globals = nlg[:,2][nlg[:,0] == k]
    k_i_global = k_globals[k_is == i]
    global_number = k_i_global[0]
    x_nlg = wezly[:,1][wezly[:,0] == global_number][0]
    y_nlg = wezly[:,2][wezly[:,0] == global_number][0]
    return x_nlg, y_nlg
    
def xryr(k:int, ksi1:int, ksi2:int, nlg:np.ndarray, wezly:np.ndarray):
    # k - nr elementu
    x, y = 0, 0
    for i in range(1,4+1):  
        x_nlg, y_nlg = xy_nlg(k, i, nlg, wezly)
        x += x_nlg*gi(i, ksi1, ksi2)
        y += y_nlg*gi(i, ksi1, ksi2)
    return x, y
    


# funkcje bikwadratowe
@njit
def q1(ksi):
    return ksi*(ksi-1)/2
@njit
def q2(ksi):
    return (1-ksi)*(1-ksi)
@njit
def q3(ksi):
    return ksi*(ksi+1)/2
@njit
def hi(i:int,ksi1,ksi2):
    assert i >0 and i <10
    match i:
        case 1:
            return q1(ksi1)*q1(ksi2)
        case 2:
            return q3(ksi1)*q1(ksi2)
        case 3:
            return q1(ksi1)*q3(ksi2)
        case 4:
            return q3(ksi1)*q3(ksi2)
        case 5:
            return q2(ksi1)*q1(ksi2)
        case 6:
            return q3(ksi1)*q2(ksi2)
        case 7:
            return q1(ksi1)*q2(ksi2)
        case 8:
            return q2(ksi1)*q3(ksi2)
        case 9:
            return q2(ksi1)*q2(ksi2)
    
