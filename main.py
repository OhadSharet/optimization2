import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse


def ex1():
    print("lets go")


def _general_iterative_method(A, b, x0, M, N, max_iterations=9999999, sig=1e-2, w=1.0):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) A=N+M
    :param b: the solution
    :param x0: first guess
    :param M: matrix in R^(n*n) A=N+M
    :param N: matrix in R^(n*n) A=N+M
    :param max_iterations: max_iterations
    :param sig: algorithm stops when the delta is smaller the sig
    :return: x^k as the solution
    """

    last_x = x0
    inv_M = np.linalg.inv(M)
    curr_iter = 0
    while curr_iter < max_iterations:
        curr_x = (1-w)*last_x + (w * inv_M) @ (b - N @ last_x)
        c = np.linalg.norm(A @ curr_x - b, 2) / np.linalg.norm(b, 2)
        print(c)
        if c < sig:
            return curr_x
        last_x = curr_x
        curr_iter += 1
    return "failed"


def Jacobi(A, b, x0, mex_iterations=100):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _general_iterative_method(A, b, x0, D, L + U, mex_iterations, sig=1e-3, w=0.35)


def Gauss_Seidel(A, b, x0, mex_iterations=300):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _general_iterative_method(A, b, x0, D + L, U, mex_iterations, sig=1e-2)


def _generate_matrix(n=256):
    A = random(n, n, 5 / n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    return A.transpose() * v * A + 0.1 * sparse.eye(n)


def test_Jacobi():
    n = 256
    b = np.random.rand(n)
    A = _generate_matrix(n)
    x_ans = Jacobi(A.toarray(), b, np.zeros(n))
    print(x_ans)


def test_Gauss_Seidel():
    n = 256
    b = np.random.rand(n)
    A = _generate_matrix(n)
    x_ans = Gauss_Seidel(A.toarray(), b, np.zeros(n))
    print(x_ans)


def ex2():
    n = 256
    A = random(n, n, 5/n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    A = A.transpose() * v * A + 0.1 * sparse.eye(n)


def ex4a():
    L = np.array([[2, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, 3, -1, 0, 0, 0, 0, 0, 0],
                  [0, 0, -1, 5, -1, 0, -1, 0, -1, -1],
                  [0, 0, 0, -1, 4, -1, -1, -1, 0, 0],
                  [0, 0, 0, 0, -1, 3, -1, -1, 0, 0],
                  [0, 0, 0, -1, -1, -1, 5, -1, 0, -1],
                  [0, 0, 0, 0, -1, -1, -1, 4, 0, -1],
                  [0, 0, 0, -1, 0, 0, 0, 0, 2, -1],
                  [0, 0, 0, -1, 0, 0, -1, -1, -1, 4]])

    b = np.array([[1],
                  [-1],
                  [1],
                  [-1],
                  [1],
                  [-1],
                  [1],
                  [-1],
                  [1],
                  [-1]])

    print(L)


if __name__ == "__main__":
    test_Gauss_Seidel()
