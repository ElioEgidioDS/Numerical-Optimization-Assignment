import numpy as np
from collections import defaultdict
from scipy.sparse import diags

class Problem_64:
    def __init__(self,n,phi):
        self.n = n
        self.phi = phi
        self.h = 1.0/(n+1)
    
    

    def function_k(self,x):
        phi = self.phi
        h = self.h
        n = self.n

        '''
        THIS IS TOO COSTLY FOR N = 1E5
        f = defaultdict(lambda:0)
        for k in range(n):
            if k == 1:
                f[k] = 2*x(k) + phi*h**2*np.sinh(phi*x(k)) - x(k+1)
            elif k == self.n:
                f[k] = 2*x(k) + phi*h**2*np.sinh(phi*x(k)) - x(k-1) -1
            else:
                f[k] = 2*x(k) + phi*h**2*np.sinh(phi*x(k)) - x(k+1) -x(k-1)
        return f'''
    
        common_term = 2 * x +(phi * h**2) * np.sinh(phi * x)

        f = np.zeros(n)

        f[1:-1] = common_term[1:-1] -x[:-2] - x[2:]
        f[0] = common_term[0] - x[1]
        f[-1] = common_term[-1] - x[-2] -1

        return f
    
    def gradient(self,x):

        f = self.function_k(x)

        n = self.n
        phi = self.phi
        h = self.h

        diagonal = 2 + (phi**2 * h**2) * np.cosh(phi * x)

        gradient = np.zeros(n)

        gradient[1:-1] = (f[1:-1] * diagonal[1:-1]) - f[:-2] - f[2:]
        gradient[0] = (f[0] * diagonal[0]) - f[1]
        gradient[-1] = (f[-1] * diagonal[-1]) - f[-2]

        return gradient
    
    def hessian(self,x):

        f = self.function_k(x)

        n = self.n
        phi = self.phi
        h = self.h

        diagonal_J = 2 + (phi**2 * h**2) * np.cosh(phi * x)
        off_diagonal_J = -1*np.ones(n-1)

        J = diags([off_diagonal_J, diagonal_J, off_diagonal_J],
                  [-1, 0, 1],
                  shape=(n,n),
                  format='csr')
        first_term = J.T @ J

        second_sub_term = (phi**3 * h**2) * np.sinh(phi * x)
        second_term_diagonal = f*second_sub_term
        second_term = diags(second_term_diagonal, 0, shape=(n,n),format='csr')

        hessian = first_term + second_term
        return hessian

    def function(self, x):
        f = self.function_k(x)
        return  0.5*np.dot(f,f)