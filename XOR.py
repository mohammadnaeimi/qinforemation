import numpy as np
import random
import math


class Bin():
    # Attributes: Real numbers or matrices

    def Clossest(self, input):
        t = []
        x = input
        if x == 0:
            return 0
        else:
            while x != 1:
                t.append(2)
                m, trash = divmod(x, 2)
                x = m
            res = 2 ** (len(t))
        return res

    def powers(self, input):
        x = input
        y = Bin().Clossest(input)
        t = []
        while y > 0:
            input = input - y
            t.append(y)
            y = Bin().Clossest(input)
        input = x
        return t

    def powers2(self, input):
        x = input
        y = Bin().Clossest(input)
        t = []
        while y > 0:
            t.append(y)
            y = y / 2
        input = x
        return t

    def to_Bin(self, input, base):
        b = int(math.floor(math.log(input, 2) + 1))
        t1 = Bin().powers(input)
        t2 = Bin().powers2(input)
        bin = []
        for i in range(len(t2)):
            if t2[i] in t1:
                bin.append(1)
            else:
                bin.append(0)
        if base == b:
            return bin
        else:
            b = base - b
            res = [0] * b
            res.extend(bin)
            return res

    def to_Dec(self, input):
        dec = 0
        t = input
        for i in range(len(t)):
            dec += 2 ** (len(t) - 1 - i) * t[i]
        return dec

    def bin_mod(self, input):
        m, n = divmod(input, 2)
        if n == 0:
            return 0
        else:
            return 1

    def bin_matrix(self, input):
        x, y = input.shape
        bin = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                x = input
                input = input[i][j]
                bin[i][j] = Bin().bin_mod(input)
                input = x
        return bin


class Matrix():

    # Attributes: Numpy matrices

    def I(self, input):
        I = np.zeros((input,input), int)
        for i in range(input):
            for j in range(input):
                if i == j:
                    I[i][j] = 1
        return I

    def minor(self, input, i, j):
        return input[np.array(range(i) + range(i + 1, input.shape[0]))[:, np.newaxis],
                     np.array(range(j) + range(j + 1, input.shape[1]))]

    def Determinant(self, input):
        det = 0
        x = input
        if len(input) == 2:
            return input[0][0] * input[1][1] - input[0][1] * input[1][0]
        else:
            for i in range(len(input)):
                input = x
                input = input.minor(0, i)
                det += (-1) ** i * x[0][i] * input.Determinant()
        return det

    def cofactor(self, input):
        if len(input) == 2:
            x = input
            det = input.Determinant()
            a = input.m[0][0]
            d = input.m[1][1]
            input.m[0][0] = d
            input.m[0][1] = -input.m[0][1]
            input.m[1][0] = -input.m[1][0]
            input.m[1][1] = a
            res = input
            input = x
            return (1.0 / det) * res
        else:
            leninput = len(input)
            M = np.zeros((leninput, leninput), int)
            for i in range(leninput):
                for j in range(leninput):
                    x = input
                    input = inputinor(i, j)
                    M[i][j] += (-1) ** (i + j) * input.Determinant()
                    input = x
        return M

    def Inverse(self, input):
        M = input.cofactor()
        Ainv = (1.000 / input.Determinant()) * M.T
        return Ainv

    def Direct_Product(self, input, other):
        m1, m2 = np.shape(input)
        n1, n2 = np.shape(other)
        res = np.zeros((m1 * n1, m2 * n2), int)
        for i in range(m1):
            for j in range(m2):
                for k in range(n1):
                    for l in range(n2):
                        res[i * n1 + k][j * n2 + l] = input[i][j] * other[k][l]
        return res

    def Dir_bin_prod(self, input, other):
        res = np.zeros((len(input) * len(other), 1),int)
        for i in range(len(input)):
            if input[i] == 0:
                for j in range(len(other)):
                    res[i * len(other) + j] = 0
            if input[i] == 1:
                for j in range(len(other)):
                    res[i * (len(other)) + j] = other[j]
        return res

    def Str_to_mat(self, input):
        if input[0] == 0:
            res = np.array(([1], [0]))
        else:
            res = np.array(([0], [1]))
        for i in range(len(input) - 1):
            if input[i + 1] == 0:
                res = Matrix().Dir_bin_prod(res, np.array(([1], [0])))
            else:
                res = Matrix().Dir_bin_prod(res, np.array(([0], [1])))
        return res

    def Mat_to_str(self, input):
        sum_check = [i for i in input]
        sum = 0
        for i in sum_check:
            sum += i
        if sum > 1:
            return False
        else:
            num_of_bits = int(np.log2(len(input)))
            strings = Matrix().all_strings(num_of_bits)
            for i in range(len(input)):
                res = Matrix().Str_to_mat(strings[i])
                equal_check = np.equal(res, input)
                t = [j for j in equal_check if j == True]
                if len(t) == len(equal_check):
                    return strings[i]

    def Mat_to_strss(self, input):
        t = []
        r = []
        num_of_bits = int(np.log2(len(input)))
        strings = Matrix().all_strings(num_of_bits)
        for i in range(len(input)):
            if input[i] == 1:
                r.append(strings[i])
                t.append(''.join(map(str, strings[i])))
        return r

    def all_strings(self, input):
        res = np.zeros((2 ** input, input), int)
        for i in range(2 ** input):
            if i == 0:
                res[i] = np.zeros((input,), int)
            else:
                res[i] = Bin().to_Bin(i, input)
        return res


def XOR(input, ancilla, control_bit, target_bit):

    # Number of bits and corresponding identify operation is built
    number_of_bits = int(np.log2(len(input)))
    I = Matrix().I(2 ** number_of_bits)

    # Target operator: The operator needed to act on the ancillas with respect to the target bit
    number_of_ancillas = len(ancilla)
    X = np.array(([0,1],[1,0]))
    target_operator = Matrix().I(2 ** target_bit)
    target_operator = Matrix().Direct_Product(target_operator, X)
    remained_I = number_of_ancillas - target_bit - 1
    if remained_I == 0:
        target_operator = target_operator
    else:
        remain = Matrix().I(2 * remained_I)
        target_operator = Matrix().Direct_Product(target_operator, remain)
    # Target acquired

    # pass operator
    pass_op = Matrix().I(2 ** (number_of_bits + number_of_ancillas))

    # Obtaining the whole Hilbert space of the system
    mat_input = input
    mat_ancilla = Matrix().Str_to_mat(ancilla)
    tot = Matrix().Dir_bin_prod(mat_input, mat_ancilla)

    # Collecting the strings of the whole system into a list
    t = Matrix().Mat_to_strss(tot)


    res = np.zeros((2 ** (number_of_bits + number_of_ancillas),1), int)

    for i in range(len(t)):
        ket = t[i]
        if ket[control_bit] == 0:
            res += np.dot(pass_op, Matrix().Str_to_mat(ket))
        else:
            res += np.dot(Matrix().Direct_Product(I, target_operator), Matrix().Str_to_mat(ket))
    return res


input1 = np.array((1,1,0,0))
input2 = np.array((0,0,1,1))
input3 = np.array((1,0,0,1))
ancilla = np.array((0,0))

y = Matrix().Str_to_mat(input1) + Matrix().Str_to_mat(input2) + Matrix().Str_to_mat(input3)

t = Gate().XOR(y, ancilla, 0, 1)
print Matrix().Mat_to_strss(t)