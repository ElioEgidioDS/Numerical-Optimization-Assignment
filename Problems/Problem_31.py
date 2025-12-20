import numpy as np
from scipy.sparse import diags

class Problem_31:
    def __init__(self, n):
        self.n = n
        # Starting point x_l = -1
        self.x0 = -1 * np.ones(n)

    def function_k(self, x):
        """Returns residual vector f(x). Correct."""
        n = self.n
        f = np.zeros(n)

        term = (3 - 2 * x) * x + 1

        f[0] = term[0] - 2 * x[1]

        f[1:-1] = term[1:-1] - 2 * x[2:] - x[:-2]

        f[-1] = term[-1] - x[-2]
                    
        return f

    def function(self, x):
        """Returns objective function F(x) = 0.5 * ||f(x)||^2."""
        f = self.function_k(x)
        return 0.5 * np.dot(f, f)

    def gradient(self, x):
        """Returns exact gradient: J.T @ f."""
        f = self.function_k(x)
        grad = np.zeros(self.n)

        J_diag = 3 - 4 * x
        grad += f * J_diag

        grad[:-1] += (-1) * f[1:] 

        grad[1:] += (-2) * f[:-1]

        return grad

    def hessian(self, x):
        """Returns exact sparse Hessian: J.T @ J + S. Correct."""
        n = self.n
        f = self.function_k(x)

        d_main = 3 - 4 * x
        
        d_low  = -1 * np.ones(n-1)
        
        d_up   = -2 * np.ones(n-1)
        
        J = diags([d_low, d_main, d_up], [-1, 0, 1], shape=(n, n), format='csr')
        
        H_gn = J.T @ J
        
        H_corr = diags(-4 * f, 0, shape=(n, n), format='csr')

        return H_gn + H_corr