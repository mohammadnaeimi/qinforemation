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
    t1=get_powers(x)
    t2=get_2powers(x)
    bin=[]
    for i in range(len(t2)):
        if t2[i] in t1:
            bin.append(1)
        else:
            bin.append(0)
    return bin
def Dec_to_Hilb(a,n):
    # This Function returns a decimal number into binary in the
    # required numbers of bits (n)
    t=Dec_to_Bin(a)
    l=n-len(t)
    tl=[]
    for i in range(l):
        tl.append(0)
    for j in range(len(t)):
        tl.append(t[j])
    return tl
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
def Cofactors(A):
    if len(A)==2:
        det=Determinant(A)
        a=A[0][0]
        d=A[1][1]
        A[0][0]=d
        A[0][1]=-A[0][1]
        A[1][0]=-A[1][0]
        A[1][1]=a
        return (1.0/det)*A
    else:
        M=np.zeros((len(A),len(A)))
        for i in range (len(A)):
            for j in range(len(A)):
                M[i][j]+=(-1)**(i+j)*Determinant(Minor(A,i,j))
    return M
def Minor(A,i,j):
    return A[np.array(range(i)+range(i+1,A.shape[0]))[:,np.newaxis],
               np.array(range(j)+range(j+1,A.shape[1]))]
def Determinant(A):
    det=0
    if len(A)==2:
        return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    else:
        for i in range((len(A))):
            X=Minor(A,0,i)
            det+=(-1)**i*A[0][i]*Determinant(X)
        return det
    return
def Inverse(A):
    M=Cofactors(A)
    Ainv=(1.000/Determinant(A))*M.T
    return Ainv
def Unitary_Check(U):
    IU=Inverse(U)
    Tconj=np.conjugate(U)
    for i in range(len(U)):
        for j in range(len(U)):
            if IU[i][j]==Tconj[i][j]:
                return True
            else:
                return False
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
    res=np.zeros((len(m)*len(n),1),int)
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
def Matrix_Decomposing(M):
    # This Function 'Prints' the elements of a 2^n matrix
    # in terms of its basis
    for i in range(len(M)):
        for j in range(len(M)):
            print Dec_to_Hilb(i,int(np.log2(len(M)))),'X', Dec_to_Hilb(j,int(np.log2(len(M)))),'=', M[i][j]
    return
def Three_Qubit_Bit_Flip_Channel(p):
    I=np.array(([1,0],[0,1]))
    psi0=(1/np.sqrt(2))*String_to_Matrix(np.array((0,0,0)))
    psi1=(1/np.sqrt(2))*String_to_Matrix(np.array((1,1,1)))
    x=np.array(([0,1],[1,0]))
    X1=Dir_matrix_prod(x,Dir_matrix_prod(I,I))
    X2=Dir_matrix_prod(Dir_matrix_prod(I,x),I)
    X3=Dir_matrix_prod(Dir_matrix_prod(I,I),x)
    ro0=np.dot(psi0,psi0.T)+np.dot(psi1,psi1.T)
    term1= np.dot(X1,np.dot(ro0,X1))
    term2= np.dot(X2,np.dot(ro0,X2))
    term3= np.dot(X3,np.dot(ro0,X3))
    Ero=(1-p)*ro0 +(p/3)*(term1+term2+term3)
    print np.trace(Ero)
    print Ero
    print Matrix_Decomposing(Ero)
def matrix_writing(a,file):
    mat = np.matrix(a)
    with file as f:
        for line in mat:
            np.savetxt(f, line, fmt='%10.5f')
        f.write('\n')
    return
def number_writing(a,file):
    with file as f:
        f.write('%g' %a)
        f.write('\n')
    return

def main(t):
    m,n=np.shape(t)
    with open('/home/ben11/Work/Research/Programing/Python/Exercises/Problems/My_own_writing_reading/Out.txt','a') as outfile:
        for i in range(m):
            for j in range(n):
                outfile.write('%g  ' %t[i][j])
                if i==m-1:
                    outfile.write('\n')
    return


t=np.array(([1,1],[0,1]))
if __name__ == '__main__':
    main(t)

