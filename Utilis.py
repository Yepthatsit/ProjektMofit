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
    return jm,im,np.array(sm)
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
def Vkmatrix(k, a, m, omega, nlg:np.ndarray, wezly:np.ndarray):
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
def Gmatrix(N:int, nlg:np.ndarray, wezly:np.ndarray, a: float):
    m = 0.067
    Delta = 0.0001
    omega = omega = 10/27211.6
    
    nlg_max = (4*N + 1)**2
    k_max = (2*N)**2
    S, H = np.zeros((nlg_max, nlg_max)), np.zeros((nlg_max, nlg_max))
    _, _, s = Sloc(a)
    s = s.reshape((9, 9))
    _, _, t = Tmatrix(m, Delta)
    t = t.reshape((9, 9))
    for k in range(k_max):
        _, _, v = Vkmatrix(k, a, m, omega, nlg, wezly)
        v = v.reshape((9, 9))
        for i1 in range(9):
            for i2 in range(9):
                nlg1 = nlg_number(k+1, i1+1, nlg, wezly)
                nlg2 = nlg_number(k+1, i2+1, nlg, wezly)
                #print(k, i1, i2, nlg1, nlg2)
                S[nlg1-1, nlg2-1] += s[i1, i2]
                H[nlg1-1, nlg2-1] += t[i1, i2] + v[i1, i2] 
    return S, H

@njit
def Gboundary(N:int, S:np.ndarray, H:np.ndarray, nlg:np.ndarray, wezly:np.ndarray):
    ksis = np.array([[-1,-1], [1, -1], [-1, 1], [1, 1],
                    [0, -1], [1, 0], [-1, 0], [0, 1],
                    [0, 0]])
    k_max = (2*N)**2
    x_max = wezly[:,2].max()
    for k in range(k_max):
        for i in range(9):
            global_number = nlg_number(k+1, i+1, nlg, wezly)
            ksi1 = ksis[i, 0]
            ksi2 = ksis[i, 1]
            x, y = xryr(k+1, ksi1, ksi2, nlg, wezly)
            if abs(x) == x_max or abs(y) == x_max:
                S[global_number-1, :] = 0
                S[:, global_number-1] = 0
                S[global_number-1, global_number-1] = 1
                
                H[global_number-1, :] = 0
                H[:, global_number-1] = 0
                H[global_number-1, global_number-1] = -1410
    return S, H