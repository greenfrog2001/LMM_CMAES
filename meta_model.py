from math import sqrt
import numpy as np
import scipy


def kernel(x):
    return (1 - x**2) ** 2


def distance(x, y, M):
    return np.linalg.norm(np.matmul(M, np.subtract(x, y)))


def create_Z_hat(z):
    Z_hat = np.array([])
    for zi in z:
        Z_hat = np.append(Z_hat, zi ** 2)

    for i in range(len(z)):
        for j in range(len(z)):
            if i != j:
                Z_hat = np.append(Z_hat, z[i] * z[j])

    for zi in z:
        Z_hat = np.append(Z_hat, zi)

    Z_hat = np.append(Z_hat, 1)

    return Z_hat


def build_metamodel(database, q, C, sigma, knn, dimension):

    # Calculate matrix M
    (D, B) = np.linalg.eig(C)
    D = np.diag(D)

    a = scipy.linalg.fractional_matrix_power(D, -1/2)
    M = np.matmul(a, np.transpose(B))
    M = M * (1/sigma)

    # sort list and take knn nearest points

    # x is under the form (point, score)
    T = sorted(database, key=lambda x: distance(np.array(x[0]), q, M))
    T = T[0:knn]

    distance_sknn = distance(T[knn-1][0], q, M)

    # Calculate W
    # TODO: can optimize by using the distance already calculated
    w_aux = np.array([])
    Y = np.array([])
    for (si, yi) in T:
        w_aux = np.append(w_aux, sqrt(
            kernel(distance(si, q, M) / distance_sknn)))

    W = np.diag(w_aux)

    # Calculate Z, Y
    Z = []

    for (si, yi) in T:

        zi = np.matmul(M, np.subtract(si, q))
        Z.append(create_Z_hat(zi))
        Y = np.append(Y, yi)

    Z = np.array(Z)
    Y = np.array(Y)

    B_hat = np.linalg.lstsq(Z, Y.T)[0]

    # construction of A
    A = np.diag(B_hat[0:dimension])
    i = 0
    j = 1
    index = dimension
    while i < dimension and j < dimension:
        if (i != j):
            A[i][j] = B_hat[index] / 2
            index += 1
        j += 1
        if (j == dimension):
            i += 1
            j = 0

    # construction of a
    a = B_hat[index: index + dimension]
    a0 = B_hat[-1]



    return a0

def is_in_database(point_to_search, database):
    for (point, score) in database:
        if (point_to_search == point).all():
            return True

    return False


def evalutae_lmm(optimizer, point, database, knn):
    return build_metamodel(database, point, optimizer._C, optimizer._sigma, knn, optimizer._n_dim)


def update_knn(knn_target, database, lsqdim):
    return math.floor(min(knn_target, math.sqrt(len(database) * lsqdim)))
