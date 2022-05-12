import numpy as np
import random


def Hamming(n, m):
    d = 0
    if len(n) == len(m):
        for i in range(len(n)):
            if n[i] != m[i]:
                d += 1
        return d
    else:
        return False
