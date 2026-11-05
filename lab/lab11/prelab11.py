import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm

#composite trapezoidal rule with n subintervals

def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_trap = h*f(xnode[0])*1/2
    for j in range(1,n):
        I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])
    return I_trap

#composite simpsons rule
def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
        # odd part
        I_simp = I_simp +4*f(xnode[2*j-1])
        # even part (interior only; x_n is added below)
        if j < nhalf:
            I_simp = I_simp+2*f(xnode[2*j])
    I_simp= I_simp + f(xnode[n])
    I_simp = h/3*I_simp
    return I_simp

