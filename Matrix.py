import numpy as np


def Id(n):
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                a[i][j] += 1
    print a
    return


def UpDig(a):
    for i in range(len(a)):
        for j in range(len(a)):
            if i > j:
                a[i][j] *= 0
    print a
    return


def LoDig(a):
    for i in range(len(a)):
        for j in range(len(a)):
            if i < j:
                a[i][j] *= 0
    print a
    return


def Bproduct_1by1(a, b):
    x = np.array(([0]))
    d = 0
    for i in a:
        for j in b:
            d = a[i][0] * b[j][0]
            y = np.append(x, d)
            x = y
    bres = np.delete(y, 0, 0)
    res = np.reshape(bres, (len(bres), 1))
    print res
    return


def DirProd2by2(a, b):
    M = np.zeros((len(a) + len(b), (len(a) + len(b))))
    for i in range(len(a) + len(b)):
        for j in range(len(a)):
            for k in range(len(b)):
                M[i][2 * j + k] += a[i][j] * b[i][k]

    return M


def Minor(A, i, j):
    return A[np.array(range(i) + range(i + 1, A.shape[0]))[:, np.newaxis],
             np.array(range(j) + range(j + 1, A.shape[1]))]


def Determinant(A):
    det = 0
    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    else:
        for i in range((len(A))):
            X = Minor(A, 0, i)
            det += (-1) ** i * A[0][i] * Determinant(X)
        return det
    return


def Identify(n):
    I = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                I[i][j] = 1
    return I


def Cofactors(A):
    if len(A) == 2:
        det = Determinant(A)
        a = A[0][0]
        d = A[1][1]
        A[0][0] = d
        A[0][1] = -A[0][1]
        A[1][0] = -A[1][0]
        A[1][1] = a
        return (1.0 / det) * A
    else:
        M = np.zeros((len(A), len(A)))
        for i in range(len(A)):
            for j in range(len(A)):
                M[i][j] += (-1) ** (i + j) * Determinant(Minor(A, i, j))
    return M


def Inverse(A):
    M = Cofactors(A)
    Ainv = (1.000 / Determinant(A)) * M.T
    return Ainv


def Matrix_Equation(A, B):
    return np.dot(Inverse(A), B)

print Cofactors(np.array(([1, 1, 1], [-1, 1, -1], [1, 2, 0])))
print Inverse(np.array(([1, 1, 1], [-1, 1, -1], [1, 2, 0])))