import numpy as np
import random
import math

# This program convert a text containing a matrix in to a numpy matrix
# with all the strings in text having the "same length"
# read:
      # the text file to be converted into the matrix
# a:
      # the number of elements in every row of the matrix
# b:
      # the number of characters in elements, must be equal for all elements



### NOT COMPLETE YET

def list_float_division(t,a,b):
    res=[]
    for i in range(a):
        str_div=''
        for j in range(b):
            str_div+=t[j]
        del t[0:b]
        res.append(float(str_div))
    print res

def text_to_matrix(file,a,b):
    with file as f:
        l=f.readlines()
    res = np.zeros((a,len(l)))
    for k in range(len(l)):
        line=l[k]
        t=[i for i in line if i !=' ']
        res=list_float_division(t,a,b,)
    return res


read= open('/home/ben11/Work/Research/Programing/Python/Exercises/Problems/Matrix_writing/to_matrix','r')
loadtext= open('/home/ben11/Work/Research/Programing/Python/Exercises/Problems/Matrix_writing/to_matrix', 'r')




#text_to_matrix(read,3,4)
print np.loadtxt(loadtext)


