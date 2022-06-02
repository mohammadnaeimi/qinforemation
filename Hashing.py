import numpy as np
import random

# Needed Functions

def Toeplitz(m,n):
    T=np.zeros((len(n),len(m)))
    if n[0]==m[0]:
        if len(n)>len(m): # Row Dominant Matrix
            for i in range(len(m)):
                T[0][i]=m[i]
                for j in range(len(m)-i):
                    T[j][i+j]=T[0][i]
            for i in range(len(n)):
                T[i][0]=n[i]
                if i<=len(n)-len(m):
                    for j in range(len(m)):
                        T[j+i][j]=T[i][0]
                else:
                    for j in range(len(n)-i):
                        T[i+j][j]=T[i][0]
        if len(n)<len(m): # Column Dominant Matrix
            for i in range(len(m)):
                T[0][i]=m[i]
                if i<=len(m)-len(n):
                    for j in range(len(n)):
                        T[j][i+j]=T[0][i]
                else:
                    for j in range(len(m)-i):
                        T[j][i+j]=T[0][i]
            for i in range(len(n)):
                    T[i][0]=n[i]
                    for j in range(len(n)-i):
                        T[j+i][j]=T[i][0]
        if len(n)==len(m):
            a=len(n)
            for i in range(a):
               T[0][i]=m[i]
               for j in range(a-i):
                    T[j][i+j]=T[0][i]
            for i in range(a):
                T[i][0]=n[i]
                for j in range(a-i):
                    T[i+j][j]=T[i][0]
        return T

    return

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

def bin_flip(x):
    if x==0:
        x=1
    else:
        x=0
    return x

def Random_Flip(n,X):
    t=[]
    for i in range(n):
        t.append(random.randint(0,len(X)-1))
    for i in range(len(t)):
        a=bin_flip(X[t[i]])
        X[t[i]]=a
    return X

def Hamming(n, m):
    d = 0
    if len(n) == len(m):
        for i in range(len(n)):
            if n[i] != m[i]:
                d += 1
        return d
    else:
        return False

m=np.array(([1],[0],[1],[0],[0],[0],[0],[1],[1],[0]))
n=np.array(([1],[0],[0],[0],[0],[1],[0]))
x=np.array(([1],[0],[0],[1],[1],[1],[0],[1],[0],[0]))

#Steps of The Home Work:
# a

#print Toeplitz(m,n)
y=bin_mtrx(np.dot(Toeplitz(m,n),x))
#print y




# b
x_f=Random_Flip(4,x)
y_f=np.dot(Toeplitz(m,n),x_f)
#print y_f
#print Hamming(y,y_f)
