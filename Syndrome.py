import numpy as np
import random
import math


def XOR(n, m, i, j):
    m[j] += n[i]
    m[j] = bin_mod(m[j])
    return m


def Syndrome(n):
    m = np.array((0, 0))
    m = XOR(n, m, 0, 1)
    m = XOR(n, m, 1, 1)
    m = XOR(n, m, 1, 0)
    m = XOR(n, m, 2, 0)
    return XOR(m, m, 0, 1)

def Syndrome_Matrix(n):
    x1 = Dir_matrix_prod(np.array(([1, 0], [0, 1])), np.array(([0, 1], [1, 0])))
    x2 = Dir_matrix_prod(np.array(([0, 1], [1, 0])), np.array(([1, 0], [0, 1])))
    res = Dir_bin_prod(np.array(([1], [0])), np.array(([1], [0])))

    if n[0] == 0:
        pass
    else:
        res = np.dot(x2, res)

    if n[1] == 0:
        pass
    else:
        res = np.dot(x2, res)

    if n[1] == 0:
        pass
    else:
        res = np.dot(x1, res)

    if n[2] == 0:
        pass
    else:
        res = np.dot(x1, res)
    return XOR(Matrix_to_String(res))
