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
    return (1-ksi)*(1+ksi)

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
@njit   
def wi(i:int):
    if i <=2:
        return (18 + np.sqrt(30))/36
    else:
        return (18 - np.sqrt(30))/36
@njit
def Pi(i):
    match i:
        case 1:
            return - np.sqrt(3/7 - 2/7*np.sqrt(6/5))
        case 2:
            return np.sqrt(3/7 - 2/7*np.sqrt(6/5))
        case 3:
            return np.sqrt(3/7 + 2/7*np.sqrt(6/5))
        case 4:
            return -np.sqrt(3/7 + 2/7*np.sqrt(6/5))
@njit
def Sloc(a):
    jm = []
    im = []
    sm = []
    for j in range(1,10):
        for i in range(1,10):
            jm.append(j)
            im.append(i)
            s = 0
            for l in range (1,5):
                for n in range(1,5):
                    s += wi(l)*wi(n)*hi(j,Pi(l),Pi(n))*hi(i,Pi(l),Pi(n))
            sm.append((a**2/4)*s)
    return jm,im,sm
@njit
def DifferentiateKsi2(func:typing.Callable,i, delta:float,ksi1,ksi2):
    return (func(i,ksi1,ksi2+delta) - func(i,ksi1,ksi2-delta))/(2*delta)
@njit    
def DifferentiateKsi1(func:typing.Callable,i, delta:float,ksi1,ksi2):
    return (func(i,ksi1+ delta,ksi2) - func(i,ksi1-delta,ksi2))/(2*delta)

@njit
def Tmatrix(m,Delta): #w jednostkach atomowych
    jm = []
    im = []
    Tloc = []
    for j in range(1,10):
        for i in range(1,10):
            jm.append(j)
            im.append(i)
            T = 0.0
            for l in range(1,5):
                for n in range(1,5):
                    T += wi(l)*wi(n)*(DifferentiateKsi2(hi,j,Delta,Pi(l),Pi(n))*DifferentiateKsi2(hi,i,Delta,Pi(l),Pi(n)) +
                    DifferentiateKsi1(hi,j,Delta,Pi(l),Pi(n))*DifferentiateKsi1(hi,i,Delta,Pi(l),Pi(n)))
            Tloc.append(T)
    return jm,im,np.array(Tloc)*1/(2*m)