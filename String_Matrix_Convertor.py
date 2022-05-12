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
    #        return i
    return res

def Matrix_to_String(n):
    sum_check = [i for i in n]
    sum = 0
    for i in sum_check:
        sum += i
    if sum > 1:
        return False
    else:
        num_of_bits = int(np.log2(len(n)))
        strings = all_strings(num_of_bits)
        for i in range(len(n)):
            res = String_to_Matrix(strings[i])
            equal_check = np.equal(res, n)
            t = [j for j in equal_check if j == True]
            if len(t) == len(equal_check):
                return strings[i]


print String_to_Matrix(np.array((0,0)))

