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
        q, r = gram_schmidt_method(matrix)
        rq = r.dot(q)
        eigen_vectors = eigen_vectors.dot(q)
        if np.allclose(rq, np.triu(rq), atol=1e-4):
            break
    eigen_values = np.diag(rq)
    sort = np.argsort(np.absolute(eigen_values))[::-1]
    return eigen_values[sort], eigen_vectors[sort]


def gram_schmidt_method(matrix):
    columns_counter = 0
    n, m = np.shape(matrix)
    q = np.empty([n, m])

    for iterable in matrix.T:
        copy = np.copy(iterable)
        for i in range(0, columns_counter):
            print i
            projection = np.dot(np.dot(q[:, i].T, iterable), q[:, i])
            copy -= projection
        e = copy / np.linalg.norm(copy)
        q[:, columns_counter] = e
        columns_counter += 1

    r = np.dot(q.T, matrix)
    return q, r


def getSingluarValuesAndEigenVectors(matrix):
    n, m = matrix.shape
    if m > n:
        transpose_mult = matrix.dot(matrix.T)
        values, vectors = getEigenValues(transpose_mult)
        v = matrix.T.dot(vectors)
        values = np.sqrt(abs(values))
        diag_values = np.diag(values)
        v = v.dot(diag_values)
        for i in range(diag_values.shape[0]):
            diag_values[i, i] = 1 / diag_values[i, i]
        return values, np.asmatrix(v.T)
    transpose_mult = matrix.T.dot(matrix)
    values, vectors = getEigenValues(transpose_mult)
    values = np.sqrt(values)
    return values, np.asmatrix(vectors)
