import numpy as np
import math

def Identify(n):
    I=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                I[i][j]=1
    print I
    return


def Matrix():
    Z=([1,0],[0,-1])
    Sz=np.array(Z)

    X=([0,1],[1,0])
    Sx=np.array(X)

    Y=([0,-1j],[1j,0])
    Sy=np.array(Y)

    I=([1,0],[0,1])
    Si=np.array(I)

    Z0=np.zeros((2,2))
    print Sy.T


def Eigen():
    z1=([1],[0])
    zval1=1.0
    z2=([0],[1])
    zval2=-1.0
    zec1=np.array(z1)
    zec2=np.array(z2)
    Z=zval1*np.dot(zec1,zec1.T)+zval2*np.dot(zec2,zec2.T)

    xval1=1.0/2
    xval2=-1.0/2
    xec1=np.array(z1)+np.array(z2)
    xec2=np.array(z1)-np.array(z2)
    X=xval1*np.dot(xec1,xec1.T)+xval2*np.dot(xec2,xec2.T)

    yval1=1.0/2j
    yval2=-1.0/2j
    yec1=np.array(z1)+np.array(z2)
    yec2=np.array(z1)-np.array(z2)
    Y=yval1*np.dot(yec1,np.conj(yec1.T))+yval2*np.dot(yec2,np.conj(yec2.T))

    print Z, '\n', X, '\n', Y