import pandas as pd
from numba import njit
def two():
    return 2

def ReadFile(FilePath)-> pd.DataFrame:
    return pd.read_csv(FilePath,sep = r"\s+", header=None)

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
    