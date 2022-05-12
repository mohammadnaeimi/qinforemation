import numpy as np

def bin_mod(a):
    m, n = divmod(a,2)
    if n==0:
        return 0
    else:
        return 1

def bin_mtrx(T):
    x,y = T.shape
    bin_T=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            bin_T[i][j]=bin_mod(T[i][j])
    return bin_T