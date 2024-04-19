# 矩阵乘法

import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[9, 2, 3, 4], [6, 5, 1, 2], [7, 2, 8, 5]])


def OneMultplt(A=A, B=B, i=0, j=0):
    out = 0
    for n in range(len(A[0])):
        out += A[i][n] * B[n][j]
    return out


def MultplyMatrix(A=A, B=B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                res[i][j] = OneMultplt(A, B, i, j)
        return res

    else:
        return "输入矩阵有误！"


print(MultplyMatrix(A, B))

# 矩阵乘法：r_ij = sum(i->n) (a_in * b_nj)
