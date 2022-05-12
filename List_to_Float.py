import numpy as np
import random
import math

# This program divides a list of strings in to a list of floats:
   # t as the list of strings
   # a the number of divisions
   # b the number of strings that is going to be devide

def list_float_division(t,a,b):
    res=[]
    for i in range(a):
        str_div=''
        for j in range(b):
            str_div+=t[j]
        del t[0:b]
        res.append(float(str_div))
    print res



t=['1','2','3','4','5','6','7','8']
list_float_division(t,2,4)
