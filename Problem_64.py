import numpy as np
from collections import defaultdict
from scipy.sparse import diags

class Problem_64:
    def __init__(self,n,rho):
        self.n = n
        self.rho = rho
        self.h = 1.0/(n+1)
    

# Returns vector f_k(x) 
    def function_k(self,x):

        rho = self.rho
        h = self.h
        n = self.n
        f = np.zeros(n)
        
        # intermediate term: common to all indexes
        term = 2 * x +(rho * h**2) * np.sinh(rho * x)

        # construction of all terms
        # k = 1 (=0 for python indexes)
        f[0] = term[0] - x[1]
             
        # 1 < k < n
        f[1:-1] = term[1:-1] - x[:-2] - x[2:]
        
        #  k = n (=n-1 for python indexes)
        f[-1] = term[-1] - x[-2] -1

        return f
    
# Returns function F(x) = 0.5 * ||f(x)||^2
    def function(self, x):
        
        f = self.function_k(x)
        return 0.5 * np.dot(f, f)



    def get_jacobian_diagonal(self, x):
        return 2 + (self.rho**2 * self.h**2) * np.cosh(self.rho * x)
    

# Returns exact gradient
    def gradient(self,x):

        f = self.function_k(x)
        n = self.n
        rho = self.rho
        h = self.h
        grad = np.zeros(n)

        # computation of d(f_k) / d(x_k)
        diag = 2 + ((rho**2) * (h**2) * np.cosh(rho * x))

        # contribution of [f_k] (for all k)
        grad += f * diag

        # contribution of [f_{k-1}] ( k > 0)
        grad[1:] += (-1) * f[:-1]

        # contribution of [f_{k+1}] ( k < n-1)
        grad[:-1] += (-1) * f[1:]

        return grad
    

# Returns exact sparse Hessian: J.T @ J + Second order terms
    def hessian(self,x):

        f = self.function_k(x)

        n = self.n
        rho = self.rho
        h = self.h

        # computation of d(f_k) / d(x_k)
        diag = 2 + ((rho**2) * (h**2) * np.cosh(rho * x))
        
        # computation of d(f_k) / d(x_{k-1}
        lower_diag = -1 * np.ones(n-1)
        
        # computation of d(f_k) / d(x_{k+1}
        upper_diag = -1 * np.ones(n-1)

        # tridiagonal matrix J
        J = diags([lower_diag, diag, upper_diag],[-1, 0, 1], shape=(n,n), format='csr')
        
        # first order terms
        first_order = J.T @ J
        
        # computation of d^2(f_k) / d(x_k)^2
        diag_2 =  ((rho**3) * (h**2) * np.sinh(rho * x)) * f
        
        # second order terms
        second_order = diags(diag_2, 0, shape=(n,n), format='csr')

        return first_order + second_order
