import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse


def ex1():
    print("lets go")


def _genetal_iterativ_method(A, b, x0, M, N, max_iterations=9999999, sig=1e-2):
    '''
    Ax = b , we need to find x
    :param A:matrix in R^(n*n) A=N+M
    :param b: the solotion
    :param x0:first guess
    :param M: matrix in R^(n*n) A=N+M
    :param N: matrix in R^(n*n) A=N+M
    :return: xk as the solution
    '''

    last_x = x0
    inv_M = np.linalg.inv(M)
    curr_iter = 0
    while curr_iter < max_iterations:
        curr_x = inv_M @ (b - N @ last_x)
        c = np.linalg.norm(A@curr_x-b,2) / np.linalg.norm(b,2)
        print(c)
        if c < sig:
            return curr_x
        last_x = curr_x
        curr_iter += 1
    return "failed"


def Jacobi(A, b, x0, mex_iterations=100):
    D = np.diag(np.diag(A))
    U = np.triu(A)-D
    L = np.tril(A)-D

    return _genetal_iterativ_method(A, b, x0, D, L + U, mex_iterations,sig=1)

def Gauss_seidel(A, b, x0, mex_iterations=300):
    D = np.diag(np.diag(A))
    U = np.triu(A)-D
    L = np.tril(A)-D

    return _genetal_iterativ_method(A, b, x0, D+L, U, mex_iterations,sig=1e-2)


def _generate_matrix(n=256):
    A = random(n, n, 5 / n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    return A.transpose() * v * A + 0.1 * sparse.eye(n)


def test_Jacobi():
    n = 256
    b = np.random.rand(n)
    A = _generate_matrix(n)
    x_ans = Jacobi(A.toarray(), b , np.zeros(n))
    print(x_ans)

def test_Gauss_seidel():
    n = 256
    b = np.random.rand(n)
    A = _generate_matrix(n)
    x_ans = Gauss_seidel(A.toarray(), b , np.zeros(n))
    print(x_ans)


if __name__ == "__main__":
    test_Gauss_seidel()
