import numpy as np
from math import copysign, hypot


def getEigenValues(matrix):
    if matrix.shape[0] == matrix.shape[1]:
        return QR(matrix)
    else:
        raise Exception("The matrix must be square to obtain eigenvalues!")


def QR(matrix):
    eigen_vectors = np.identity(matrix.shape[0])

    for i in range(100):
        print i
        q, r = householder_reflection(matrix)
        rq = r.dot(q)
        eigen_vectors = eigen_vectors.dot(q)
        if np.allclose(rq, np.triu(rq), atol=1e-4):
            break
    eigen_values = np.diag(rq)
    sort = np.argsort(np.absolute(eigen_values))[::-1]
    return eigen_values[sort], eigen_vectors[sort]


def gram_schmidt_method(matrix):
    matrix = np.matrix(matrix)

    columns_counter = 0
    n, m = np.shape(matrix)
    q = np.empty([n, n])

    for iterable in matrix.T:
        copy = np.copy(iterable)
        for i in range(0, columns_counter):
            line = np.asmatrix(q[:, i])
            line_t = np.asmatrix(q[:, i].T)
            projection = np.dot(np.dot(line_t, iterable.T), line)
            copy -= projection
        e = copy / np.linalg.norm(copy)
        q[:, columns_counter] = e
        columns_counter += 1

    r = np.dot(q.T, matrix)
    return q, r

def householder_reflection(matrix):
    n, m = np.shape(matrix)
    r = np.copy(matrix)
    q = np.identity(n)
    for iter in range(n-1):
        x = r[iter:, iter]
        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -matrix[iter, iter])
        u = x + e
        v = u / np.linalg.norm(u)
        counter = np.identity(n)
        counter[iter:, iter:] -= 2.0 * np.outer(v, v)
        q = np.dot(q, counter.T)
        r = np.dot(counter, r)

    return q, r

def getSingluarValuesAndEigenVectors(A):
    m,n = A.shape
    if n > m:
        aux = A.dot(A.T)
        S, U = getEigenValues(aux)
        S = np.sqrt(abs(S))
        V = A.T.dot(U)
        S1 = np.diag(S)
        for k in range(S1.shape[0]):
            S1[k,k] = 1/S1[k,k]

        V = V.dot(S1)
        return S, np.asmatrix(V.T)
    aux = A.T.dot(A)
    S,V = getEigenValues(aux)
    S = np.sqrt(S)
    return S, np.asmatrix(V)