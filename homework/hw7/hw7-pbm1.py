import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt


def driver():

    f = lambda x: 1 / (1 + (10*x)**2)

    a = -1
    b = 1

    ''' fine grid for plotting '''
    Neval = 1001
    xeval = np.linspace(a, b, Neval)
    fex = f(xeval)

    ''' try several values of N (number of interpolation points) '''
    for N in [2, 3, 4, 5, 8, 11, 14, 17, 18, 19, 20]:

        ''' interpolation nodes x_i = -1 + (i-1)h, h = 2/(N-1) '''
        h = 2 / (N - 1)
        xint = a + np.arange(N) * h      
        yint = f(xint)

        ''' vandermonde matrix and coefficients '''
        V = Vandermonde(xint, N)
        coef = inv(V) @ yint             # coef[0] = c_1, ..., coef[N-1] = c_N

        ''' evaluate the polynomial on the fine grid '''
        peval = eval_monomial(xeval, coef)

        print(f'N = {N:2d}   max |p(x)| = {np.max(np.abs(peval)):10.4f}')

        ''' plot data, polynomial, and f on the same figure '''
        plt.figure()
        plt.plot(xint, yint, 'o', label='data points')
        plt.plot(xeval, peval, label='interpolating polynomial p(x)')
        plt.plot(xeval, fex, '--', label='f(x)')
        plt.title(f'Monomial interpolation, N = {N}')
        plt.xlabel('x')
        plt.legend()

    plt.show()
    return


def eval_monomial(xeval, coef):
    yeval = np.zeros_like(xeval)
    for c in coef:
        yeval = yeval * xeval + c
    return yeval


def Vandermonde(xint, N):
    ''' row j is [x_j^(N-1), x_j^(N-2), ..., x_j, 1] '''
    V = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            V[j][k] = xint[j]**(N - 1 - k)
    return V


if __name__ == '__main__':
    driver()