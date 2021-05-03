import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse


def ex1b():
    n = 256
    A = random(n, n, 5/n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    A = A.transpose() * v * A + 0.1 * sparse.eye(n)


def _General_Iterative_Method(A, b, x0, M, N, max_iterations=9999999, sig=1e-2, w=1.0):
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


def Jacobi(A, b, x0, max_iterations=100):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _General_Iterative_Method(A, b, x0, D, L + U, max_iterations, sig=1e-3, w=0.35)


def Gauss_Seidel(A, b, x0, max_iterations=300):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _General_Iterative_Method(A, b, x0, D + L, U, max_iterations, sig=1e-2)


def Steepest_Descent(A, b, x0, max_iterations=9999999, sig=1e-2):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) SPD
    :param b: the solution
    :param x0: first guess
    :param max_iterations: max_iterations
    :param sig: algorithm stops when the delta is smaller the sig
    :return: x^k as the solution
    """

    last_x = x0
    last_r = b - A @ x0
    curr_iter = 0
    while curr_iter < max_iterations:
        Ar = A @ last_r
        alpha = (last_r.transpose() @ last_r) / (last_r.transpose() @ Ar)
        curr_x = last_x + alpha * last_r
        curr_r = last_r - alpha * Ar
        c = np.linalg.norm(A @ curr_x - b, 2) / np.linalg.norm(b, 2)
        print(c)
        if c < sig:
            return curr_x
        last_x = curr_x
        last_r = curr_r
        curr_iter += 1
    return "failed"


def Conjugate_Gradient(A, b, x0, max_iterations=9999999, sig=1e-2):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) SPD
    :param b: the solution
    :param x0: first guess
    :param max_iterations: max_iterations
    :param sig: algorithm stops when the delta is smaller the sig
    :return: x^k as the solution
    """

    last_x = x0
    last_r = b - A @ x0
    last_p = last_r
    curr_iter = 0
    while curr_iter < max_iterations:
        Ap = A @ last_p
        alpha = (last_r.transpose() @ last_r) / (last_p.transpose() @ Ap)
        curr_x = last_x + alpha * last_p
        curr_r = last_r - alpha * Ap
        c = np.linalg.norm(A @ curr_x - b, 2) / np.linalg.norm(b, 2)
        print(c)
        if c < sig:
            return curr_x
        beta = (curr_r.transpose() @ curr_r) / (last_r.transpose() @ last_r)
        last_p = curr_r + beta * last_p
        last_x = curr_x
        last_r = curr_r
        curr_iter += 1
    return "failed"


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


def ex3c():
    print("ex 3")
    A = np.array([[5, 4, 4, -1, 0],
                  [3, 12, 4, -5, -5],
                  [-4, 2, 6, 0, 3],
                  [4, 5, -7, 10, 2],
                  [1, 2, 5, 3, 10]])
    b = np.array([[1],
                  [1],
                  [1],
                  [1],
                  [1]])
    x0 = np.array([[0],
                   [0],
                   [0],
                   [0],
                   [0]])
    GMRES_1(A, b, x0)


def GMRES_1(A, b, x0, max_iterations=50):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) SPD
    :param b: the solution
    :param x0: first guess
    :param max_iterations: max_iterations
    :return: x^k as the solution
    """

    last_x = x0
    curr_x = last_x
    last_r = b - A @ x0
    curr_iter = 0
    while curr_iter < max_iterations:
        Ar = A @ last_r
        alpha = (last_r.transpose() @ Ar) / (last_r.transpose() @ A.transpose() @ Ar)
        curr_x = last_x + alpha * last_r
        curr_r = last_r - alpha * Ar
        c = np.linalg.norm(A @ curr_x - b, 2) / np.linalg.norm(b, 2)
        print(c)
        last_x = curr_x
        last_r = curr_r
        curr_iter += 1
    print("iteration: " + str(curr_iter))
    return curr_x


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
    x = Jacobi_Standard(L, b, np.zeros(10))
    print(x)


def Jacobi_Standard(A, b, x0):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _General_Iterative_Method(A, b, x0, D, L + U, max_iterations=9999999, sig=1e-5, w=0.35)


if __name__ == "__main__":
    #test_Jacobi()
    #test_Gauss_Seidel()
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

    x0 = np.array([[1],
                  [1],
                  [1],
                  [1],
                  [1],
                  [1],
                  [1],
                  [1],
                  [1],
                  [1]])

    #print(Steepest_Descent(L, b, x0))
    #print(Conjugate_Gradient(L, b, x0))
    #ex3c()
    ex4a()
