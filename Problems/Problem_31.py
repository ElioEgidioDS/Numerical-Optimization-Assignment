import numpy as np
from scipy.sparse import diags

class Problem_31:
    def __init__(self, n):
        self.n = n
        self.name = "p31"
        # Starting point x_l = -1
        self.x0 = -1 * np.ones(n)


# Returns vector f_k(x) 
    def function_k(self, x):
       
        n = self.n
        f = np.zeros(n)

        # intermediate term: common to all indexes
        term = (3 - 2 * x) * x + 1

        # construction of all terms
        # k = 1 (=0 for python indexes)
        f[0] = term[0] - 2 * x[1]

        # 1 < k < n
        f[1:-1] = term[1:-1] - 2 * x[2:] - x[:-2]
        
        #  k = n (=n-1 for python indexes)
        f[-1] = term[-1] - x[-2]
                    
        return f


# Returns function F(x) = 0.5 * ||f(x)||^2
    def function(self, x):

        f = self.function_k(x)
        return 0.5 * np.dot(f, f)


# Returns exact gradient
    def gradient(self, x):

        n = self.n
        f = self.function_k(x)
        grad = np.zeros(n)

        # computation of d(f_k) / d(x_k)
        diag = 3 - 4 * x

        # contribution of [f_k] (for all k)
        grad += f * diag

        # contribution of [f_{k+1}] ( k < n)
        grad[:-1] += (-2) * f[1:] 
        
        # contribution of [f_{k-1}] (k > 0)
        grad[1:] += (-1) * f[:-1]

        return grad


# Returns exact sparse Hessian: J.T @ J + Second order terms
    def hessian(self, x):

        n = self.n
        f = self.function_k(x)
        
        # computation of d(f_k) / d(x_k)
        main_diag = 3 - 4 * x
        
        # computation of d(f_k) / d(x_{k-1}
        lower_diag  = -1 * np.ones(n-1)
        
        # computation of d(f_k) / d(x_{k+1}
        upper_diag   = -2 * np.ones(n-1)
        
        # tridiagonal matrix J
        J = diags([lower_diag, main_diag, upper_diag],[-1, 0, 1], shape=(n,n), format='csr')
        
        # first order terms
        first_order = J.T @ J
        
        # computation of d^2(f_k) / d(x_k)^2
        diag_2 = -4 * f
        
        # second order terms
        second_order = diags(diag_2, 0, shape=(n, n), format='csr')

        return first_order + second_order