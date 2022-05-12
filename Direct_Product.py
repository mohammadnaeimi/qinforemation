import numpy as np
import random


def Dir_bin_prod(m,n):
    res=np.zeros((len(m)*len(n),1))
    for i in range(len(m)):
        if m[i]==0:
            for j in range(len(n)):
                res[i*len(n)+j]=0
        if m[i]==1:
            for j in range(len(n)):
                res[i*(len(n))+j]=n[j]
    return res

def Dir_matrix_prod(m,n):
    m1,m2=np.shape(m)
    n1,n2=np.shape(n)
    res=np.zeros((m1*n1,m2*n2),int)
    for i in range(m1):
        for j in range(m2):
            for k in range(n1):
                for l in range(n2):
                    res[i*n1+k][j*n2+l]=m[i][j]*n[k][l]
    return res

x= np.array(([0,1],[0,1]))
print x
y=np.array(([1,2,-1],[1,3,2],[4,2,1]))
print y
print Dir_matrix_prod(x,y)