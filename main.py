import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def ex1b():
    n = 256
    A = _generate_matrix(n)
    b = np.random.rand(n, 1)
    x0 = np.array(np.zeros((n, 1), dtype=int))
    test_Jacobi(A, b, x0)
    #test_Gauss_Seidel(A, b, x0)
    #test_Steepest_Descent(A, b, x0)
    #test_Conjugate_Gradient(A, b, x0)


def _generate_matrix(n=256):
    A = random(n, n, 5 / n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    return A.transpose() * v * A + 0.1 * sparse.eye(n)


def test_Jacobi(A, b, x0):
    x_ans = Jacobi(A.toarray(), b, x0)
    print("Jacobi x: \n %s" % x_ans)


def test_Gauss_Seidel(A, b, x0):
    x_ans = Gauss_Seidel(A.toarray(), b, x0)
    print("Gauss_Seidel x: \n %s" % x_ans)


def test_Steepest_Descent(A, b, x0):
    x_ans = Steepest_Descent(A.toarray(), b, x0)
    print("Steepest_Descent x: \n %s" % x_ans)


def test_Conjugate_Gradient(A, b, x0):
    x_ans = Conjugate_Gradient(A.toarray(), b, x0)
    print("Conjugate_Gradient x: \n %s" % x_ans)


def print_graph(data_queue, total_iterations):
    x = []
    y = []
    for i in range(total_iterations + 1):
        x.append(i)
        y.append(data_queue.pop(0))
    fig, ax = plt.subplots()
    ax.semilogy(x, y)
    ax.set_title("residual")
    plt.show()


def _General_Iterative_Method(A, b, x0, M, N, max_iterations=99999, sigma=1e-2, w=1.0):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) A=N+M
    :param b: the solution
    :param x0: first guess
    :param M: matrix in R^(n*n) A=N+M
    :param N: matrix in R^(n*n) A=N+M
    :param max_iterations: max_iterations
    :param sigma: algorithm stops when the delta is smaller the sigma
    :return: x^k as the solution
    """

    residual_queue = []
    converges_queue = []
    last_x = x0
    M_inverse = np.linalg.inv(M)
    curr_iter = 0
    while curr_iter < max_iterations:
        curr_x = (1-w)*last_x + (w * M_inverse) @ (b - N @ last_x)
        c = np.linalg.norm(A @ curr_x - b, 2) / np.linalg.norm(b, 2)
        residual_queue.append(c)
        print(c)
        if c < sigma:
            print_graph(residual_queue, curr_iter)
            return curr_x
        last_x = curr_x
        curr_iter += 1
    return "failed"


def Jacobi(A, b, x0, max_iterations=99999, sigma=1e-3, w=0.35):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _General_Iterative_Method(A, b, x0, D, L + U, max_iterations, sigma, w)


def Gauss_Seidel(A, b, x0, max_iterations=99999, sigma=1e-2):
    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    return _General_Iterative_Method(A, b, x0, D + L, U, max_iterations, sigma)


def Steepest_Descent(A, b, x0, max_iterations=99999, sigma=1e-2):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) SPD
    :param b: the solution
    :param x0: first guess
    :param max_iterations: max_iterations
    :param sigma: algorithm stops when the delta is smaller the sigma
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
        if c < sigma:
            return curr_x
        last_x = curr_x
        last_r = curr_r
        curr_iter += 1
    return "failed"


def Conjugate_Gradient(A, b, x0, max_iterations=99999, sigma=1e-2):
    """
    Ax = b , we need to find x
    :param A: matrix in R^(n*n) SPD
    :param b: the solution
    :param x0: first guess
    :param max_iterations: max_iterations
    :param sigma: algorithm stops when the delta is smaller the sigma
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
        if c < sigma:
            return curr_x
        beta = (curr_r.transpose() @ curr_r) / (last_r.transpose() @ last_r)
        last_p = curr_r + beta * last_p
        last_x = curr_x
        last_r = curr_r
        curr_iter += 1
    return "failed"


def ex3c():
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
    x0 = np.array(np.zeros((5, 1), dtype=int))
    x = GMRES_1(A, b, x0)
    print("x: \n %s" % x)


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
        alpha = (last_r.transpose() @ Ar) / (Ar.transpose() @ Ar)
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
    x0 = np.array(np.zeros((10, 1), dtype=int))

    x = Jacobi(L, b, x0, max_iterations=99999, sigma=1e-5, w=1.0)
    print("x: \n %s" % x)


def ex4b():
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
    x0 = np.array(np.zeros((10, 1), dtype=int))

    M1 = L[0:3, 0:3]
    M2 = L[3:10, 3:10]
    M = np.block([[M1, np.zeros((3, 7))],
                  [np.zeros((7, 3)), M2]])
    N = L - M

    x = _General_Iterative_Method(L, b, x0, M, N, max_iterations=99999, sigma=1e-5, w=0.7)
    print("x: \n %s" % x)


def ex4c():
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
    x0 = np.array(np.zeros((10, 1), dtype=int))

    # Swapping L Rows 4,8 and Columns 4,8
    L_Swap = L
    L_Swap[[3, 7]] = L_Swap[[7, 3]]
    L_Swap[:, [3, 7]] = L_Swap[:, [7, 3]]

    M1 = L_Swap[0:3, 0:3]
    M2 = L_Swap[3:7, 3:7]
    M3 = L_Swap[7:10, 7:10]
    M = np.block([[M1, np.zeros((3, 7))],
                  [np.zeros((4, 3)), M2, np.zeros((4, 3))],
                  [np.zeros((3, 7)), M3]])
    N = L_Swap - M

    x = _General_Iterative_Method(L_Swap, b, x0, M, N, max_iterations=99999, sigma=1e-5, w=0.7)
    print("x: \n %s" % x)


if __name__ == "__main__":
    ex1b()
    #ex3c()
    #ex4a()
    #ex4b()
    #ex4c()
