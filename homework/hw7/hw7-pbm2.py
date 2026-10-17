import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


#    f = lambda x: np.exp(x)
    f = lambda x: 1/(1+(10*x)**2)


    N = 20
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = f(xint)

    '''barycentric weights'''
    w = bary_weights(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1) # lagrange
    yeval_b = np.zeros(Neval+1) # barycentric
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_b[kk] = eval_bary(xeval[kk], xint, yint, w)

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, yeval_l, 'bs--', label='lagrange')
    plt.plot(xeval, yeval_b, 'g:', label='barycentric')
    plt.legend()

    plt.figure()
    plt.semilogy(xeval, abs(yeval_l - fex), 'ro--', label='lagrange')
    plt.semilogy(xeval, abs(yeval_b - fex), 'g:', label='barycentric')
    plt.legend()
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
def bary_weights(xint):
    N1 = len(xint)
    w = np.ones(N1)
    for j in range(N1):
        for k in range(N1):
            if k != j:
                w[j] /= (xint[j] - xint[k])
    return w

def eval_bary(xeval, xint, yint, w):
    diff = xeval - xint
    if np.any(diff == 0):                 # x lands exactly on a node
        return yint[np.argmin(np.abs(diff))]
    t = w / diff
    return np.sum(t * yint) / np.sum(t)

driver()        
