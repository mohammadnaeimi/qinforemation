import numpy as np
import random

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

def XOR(n,m,i,j):
    m[j]+=n[i]
    m[j]=bin_mod(m[j])
    return m


#2by2 CNOT
CNOT=np.array(([1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]))
#print String_to_Matrix(np.array((1,1)))
#print np.dot(CNOT,String_to_Matrix(np.array((1,1))))

a=np.array(([0],[1]))
b=np.array(([1],[1]))
print Controlled_Not(a,b,0)
