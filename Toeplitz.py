import numpy as np
import random

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