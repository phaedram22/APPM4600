import numpy as np

def driver():

# test functions
    f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....

    f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...

    Nmax = 100
    tol = 1e-6

# test f1 '''
    x0 = 0.0
    [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)

#test f2 '''
    x0 = 0.0
    [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f2(xstar))
    print('Error message reads:',ier)

# test the vector version on f1
    x0 = 0.0
    [xstar,ier,x_vec] = fixedpt_vec(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    print('All iterates:',x_vec)

# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, ier]

def fixedpt_vec(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    ''' returns [xstar, ier, x_vec] where x_vec holds every iterate,
        including x0, in order'''

    # allocate +1 to leave room for x0 itself
    x_vec = np.zeros(Nmax+1)
    x_vec[0] = x0

    count = 0
    while (count < Nmax):
        count = count + 1
        x1 = f(x0)
        x_vec[count] = x1
        if (abs(x1-x0) < tol):
            xstar = x1
            ier = 0
            x_vec = x_vec[:count+1]   # trim off unused entries
            return [xstar, ier, x_vec]
        x0 = x1

    xstar = x1
    ier = 1
    x_vec = x_vec[:count+1]
    return [xstar, ier, x_vec]

driver()