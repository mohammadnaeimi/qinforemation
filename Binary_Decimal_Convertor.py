import random

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
    t1=get_powers(x)
    t2=get_2powers(x)
    bin=[]
    for i in range(len(t2)):
        if t2[i] in t1:
            bin.append(1)
        else:
            bin.append(0)
    return bin

def Bin_to_Dec(t):
    dec=0
    for i in range(len(t)):
        dec+=2**(len(t)-1-i)*t[i]
    return dec

def Bin_add(x,y):
    xdec=Bin_to_Dec(x)
    ydec=Bin_to_Dec(y)
    zdec=xdec+ydec
    zbin=Dec_to_Bin(zdec)
    return zbin

def Bin_sub(x,y):
    xdec=Bin_to_Dec(x)
    ydec=Bin_to_Dec(y)
    zdec=xdec-ydec
    zbin=Dec_to_Bin(zdec)
    return zbin

def Bin_mol(x,y):
    xdec=Bin_to_Dec(x)
    ydec=Bin_to_Dec(y)
    zdec=xdec*ydec
    zbin=Dec_to_Bin(zdec)
    return zbin

def Bin_div(x,y):
    xdec=Bin_to_Dec(x)
    ydec=Bin_to_Dec(y)
    zdec,r=divmod(xdec,ydec)
    zbin=Dec_to_Bin(zdec)
    return zbin


print Dec_to_Bin(0)
