import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():

    x0 = np.array([0.5, 0.5, 0.5])

    Nmax = 100
    sd_tol = 5e-2
    tol = 1e-6

    t = time.time()
    for j in range(50):
        [x_sd, gval, ier_sd] = SteepestDescent(x0, sd_tol, Nmax)
        [xstar, ier, its] = Newton(x_sd, tol, Nmax)
    elapsed = time.time() - t

    print('Steepest descent result:', x_sd)
    print('Steepest descent g value:', gval, ' ier:', ier_sd)
    print('Newton result:', xstar)
    print('Newton: the error message reads:', ier)
    print('Newton: took this many seconds:', elapsed/50)
    print('Newton: number of iterations is:', its)

def evalF(x):
# vector valued function you are trying to find the root of 
    F = np.zeros(3)
    F[0] = x[0] + np.cos(x[0]*x[1]*x[2]) - 1
    F[1] = (1-x[0])**(1/4) + x[1] + 0.5*x[2]**2 - 0.15*x[2] - 1
    F[2] = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
    return F

def evalJ(x):
    xs, y, z = x[0], x[1], x[2]
    s = np.sin(xs*y*z)

    J = np.array([
        [1 - y*z*s,   -xs*z*s,       -xs*y*s],
        [-0.25*(1 - xs)**(-3/4),  1,  z - 0.15],
        [-2*xs,       -0.2*y + 0.01,  1]])
    return J

def evalg(x):
# The least squares function you are minimizing

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
# Grad g
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)
        
# below is finding the step-length using the quadratic min. 
        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3
#####
        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]

def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

driver()