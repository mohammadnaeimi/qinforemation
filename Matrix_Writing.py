import numpy as np
import random
import math

def matrix_writing(a,file):
    mat = np.matrix(a)
    with file as f:
        for line in mat:
            np.savetxt(f, line, fmt='%10.5f')
        f.write('\n')
    return

def number_writing(a,file):
    with file as f:
        f.write('%g\n' %a)
    return

a=np.array(([1,1,2],[0,1,-1],[-1,0,1],[1,-1,2]))

file= open('/home/ben11/Work/Research/Programing/Python/Exercises/Problems/Matrix_writing/matrix1','wb+')

matrix_writing(a,file)
