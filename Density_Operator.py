import numpy as np
import random
import math

def Density_Operator(psi):
    return np.dot(psi,psi.T)

def Random_State_Matrix_Form(n):
    state=np.random.random_sample((n,1))
    norm=0
    for i in range(n):
        norm+=state[i]**2
    normal=np.sqrt(norm)
    return (1.0/normal)*state

psi= Random_State_Matrix_Form(2)
ro= Density_Operator(psi)
print ro
print np.trace(ro)
print np.dot(ro,ro)
