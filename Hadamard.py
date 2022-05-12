import numpy as np
import random
import math

def closest(x):
    t=[]
    if x==0:
        return 0
    else:
        while x != 1:
            t.append(2)
            m, y = divmod(x, 2)
            x = m
        s = 2 ** len(t)
        return s

def get_powers(x):
    y=closest(x)
    t=[]
    while y>0:
        x=x-y
        t.append(y)
        y=closest(x)
    return t

def get_2powers(x):
    y=closest(x)
    t=[]
    while y>0:
        t.append(y)
        y=y/2
    return t

def Dec_to_Bin(x):
    lenstr=int(math.floor(math.log(x,2)+1))
    t1=get_powers(x)
    t2=get_2powers(x)
    bin=np.zeros((len(t2),),int)
    for i in range(len(t2)):
        if t2[i] in t1:
            bin[i]=1
        else:
            bin[i]=0
    return bin

def Dir_bin_prod(m,n):
    res=np.zeros((len(m)*len(n),1))
    for i in range(len(m)):
        for l in range(len(n)):
            res[i*(len(n))+l]=m[i]*n[l]
    return res

def Dir_matrix_prod(m,n):
    m1,m2=np.shape(m)
    n1,n2=np.shape(n)
    res=np.zeros((m1*n1,m2*n2))
    for i in range(m1):
        for j in range(m2):
            for k in range(n1):
                for l in range(n2):
                    res[i*n1+k][j*n2+l]=m[i][j]*n[k][l]
    return res

def String_to_Matrix(n):
    if n[0]==0:
        res=np.array(([1],[0]))
    else:
        res=np.array(([0],[1]))
    for i in range(len(n)-1):
        if n[i+1]==0:
            res=Dir_bin_prod(res,np.array(([1],[0])))
        else:
            res=Dir_bin_prod(res,np.array(([0],[1])))
    #for i in range(len(res)):
    #    if res[i]==1:
    #        indx=i
    #        print i
    return res

def all_strings(n):
    res=np.zeros((2**n,n),int)
    for i in range(2**n):
        if i==0:
            res[i]= np.zeros((n,),int)
        else:
            x= Dec_to_Bin(i)
            if len(x)<n:
                l=n-len(x)
                zeroesadded=np.zeros((l,),int)
                res[i]= np.append(zeroesadded,x)
            else:
                res[i]= x
    return res

def Hadamard_gate(n):
    # The Argument is a String not a Column Matrix,
    # eventhough it could be done by a Column Matrix
    # This Function applies a Hadamard gate on a given string and return the result
    fac=math.sqrt(2**len(n))
    if n[0]==0:
        res= np.array(([1],[1]))
    else:
        res= np.array(([1],[-1]))
    for i in range(len(n)-1):
        if n[i+1]==0:
            res=Dir_bin_prod(res,np.array(([1],[1])))
        else:
            res=Dir_bin_prod(res,np.array(([1],[-1])))
    return (1/fac)*res

def Hadamard_Matrix(n):
    H=np.array(([1,1],[1,-1]))
    res=H
    for i in range(n-1):
        res=Dir_matrix_prod(res,H)
    return res

H=np.array(([1,1],[1,-1]))
m=np.array(([1],[0],[1],[0]))
n=np.array(([1],[0],[1]))
M=np.array(([1,0],[1,-1]))
N=np.array(([1,0,1],[0,1,1]))
x=np.array(([1],[1]))
y=np.array(([1],[-1]))
n=np.array((1,0,1,1))


res = np.dot(Hadamard_Matrix(4),String_to_Matrix(n))
#print res
#print all_strings(4)

HG= Hadamard_gate(np.array((1,1)))