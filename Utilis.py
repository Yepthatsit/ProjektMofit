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
def nlg_number(k:int, i:int, nlg:np.ndarray, wezly:np.ndarray):
    k_is = nlg[:,1][nlg[:,0] == k]
    k_globals = nlg[:,2][nlg[:,0] == k]
    k_i_global = k_globals[k_is == i]
    global_number = k_i_global[0]
    return global_number
        
@njit
def xy_nlg(k:int, i:int, nlg:np.ndarray, wezly:np.ndarray):
    global_number = nlg_number(k, i, nlg, wezly)
    x_nlg = wezly[:,1][wezly[:,0] == global_number][0]
    y_nlg = wezly[:,2][wezly[:,0] == global_number][0]
    #print('xd?', x_nlg, y_nlg)
    return x_nlg, y_nlg

@njit    
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

@njit
def Vkmatrix(k,a,m,omega,nlg:np.ndarray, wezly:np.ndarray):
    c = (a**2)*m*(omega**2)/8
    jm = []
    im = []
    Vloc = []
    for j in range(1,10):
        for i in range(1,10):
            V = 0.0
            for l in range(1,5):
                for n in range(1,5):
                    x,y = xryr(k,Pi(l),Pi(n),nlg,wezly)
                    x /= 0.05292
                    y /= 0.05292
                    V += wi(l)*wi(n)*hi(j,Pi(l),Pi(n))*hi(i,Pi(l),Pi(n))*((x)**2+y**2)
            jm.append(j)
            im.append(i)
            Vloc.append(V)
    return jm,im,np.array(Vloc)*c

@njit
def Gmatrix(s: np.ndarray, t: np.ndarray, v: np.ndarray, N):
    nlg_max = 4*N + 1
    k_max = N**2
    S, H = np.zeros(nlg_max, nlg_max), np.zeros(nlg_max, nlg_max)
    for k in range(k_max):
        for i1 in range(9):
            for i2 in range(9):
                pass#S[xy_nlg(k, i1, nlg, wezly)]