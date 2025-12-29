import numpy as np
from scipy.sparse import diags

class Problem_52:
    def __init__(self, n):
        self.n = n
        self.name = "p52"
        # Starting point consigliato x0 = 0
        self.x0 = np.zeros(n)

    def _safe_exp(self, arg):
        """Previene l'overflow limitando l'esponente nell'intervallo sicuro per float64."""
        return np.exp(np.clip(arg, -100, 100))

    def function_k(self, x):
        n = self.n
        f = np.zeros(n)
        
        # k = 1 (indice 0)
        f[0] = 3.0 * x[0]**3 + 2.0 * x[1] - 5.0 + np.sin(x[0] - x[1]) * np.sin(x[0] + x[1])
        
        # 1 < k < n
        if n > 2:
            # Termine esponenziale protetto
            exp_term = self._safe_exp(x[:-2] - x[1:-1])
            f[1:-1] = -x[:-2] * exp_term + x[1:-1] * (4.0 + 3.0 * x[1:-1]**2) + \
                      2.0 * x[2:] + np.sin(x[1:-1] - x[2:]) * np.sin(x[1:-1] + x[2:]) - 8.0
        
        # k = n
        exp_term_last = self._safe_exp(x[-2] - x[-1])
        f[-1] = -x[-2] * exp_term_last + 4.0 * x[-1] - 3.0
                    
        return f

    def function(self, x):
        f = self.function_k(x)
        return 0.5 * np.dot(f, f)

    def gradient(self, x):
        n = self.n
        f = self.function_k(x)
        grad = np.zeros(n)
        
        # d(f_k)/d(x_k)
        diag_J = np.zeros(n)
        diag_J[0] = 9.0 * x[0]**2 + np.cos(x[0]-x[1])*np.sin(x[0]+x[1]) + np.sin(x[0]-x[1])*np.cos(x[0]+x[1])
        if n > 2:
            # Uso lo stesso safe_exp usato in function_k
            diag_J[1:-1] = x[:-2] * self._safe_exp(x[:-2]-x[1:-1]) + (4.0 + 9.0*x[1:-1]**2) + \
                           np.cos(x[1:-1]-x[2:])*np.sin(x[1:-1]+x[2:]) + np.sin(x[1:-1]-x[2:])*np.cos(x[1:-1]+x[2:])
        diag_J[-1] = 4.0 + x[-2] * self._safe_exp(x[-2]-x[-1])

        # d(f_k)/d(x_{k-1})
        low_J = - (1.0 + x[:-1]) * self._safe_exp(x[:-1] - x[1:])
        
        # d(f_k)/d(x_{k+1})
        up_J = np.zeros(n-1)
        up_J[0] = 2.0 - np.cos(x[0]-x[1])*np.sin(x[0]+x[1]) + np.sin(x[0]-x[1])*np.cos(x[0]+x[1])
        if n > 2:
            up_J[1:] = 2.0 - np.cos(x[1:-1]-x[2:])*np.sin(x[1:-1]+x[2:]) + np.sin(x[1:-1]-x[2:])*np.cos(x[1:-1]+x[2:])

        # Grad = J.T @ f
        grad += f * diag_J
        grad[:-1] += low_J * f[1:] 
        grad[1:] += up_J * f[:-1] 

        return grad

    def hessian(self, x):
        n = self.n
        f = self.function_k(x)
        
        diag_J = np.zeros(n)
        diag_J[0] = 9.0 * x[0]**2 + np.cos(x[0]-x[1])*np.sin(x[0]+x[1]) + np.sin(x[0]-x[1])*np.cos(x[0]+x[1])
        if n > 2:
            diag_J[1:-1] = x[:-2] * self._safe_exp(x[:-2]-x[1:-1]) + (4.0 + 9.0*x[1:-1]**2) + \
                           np.cos(x[1:-1]-x[2:])*np.sin(x[1:-1]+x[2:]) + np.sin(x[1:-1]-x[2:])*np.cos(x[1:-1]+x[2:])
        diag_J[-1] = 4.0 + x[-2] * self._safe_exp(x[-2]-x[-1])

        low_J = - (1.0 + x[:-1]) * self._safe_exp(x[:-1] - x[1:])
        
        up_J = np.zeros(n-1)
        up_J[0] = 2.0 - np.cos(x[0]-x[1])*np.sin(x[0]+x[1]) + np.sin(x[0]-x[1])*np.cos(x[0]+x[1])
        if n > 2:
            up_J[1:] = 2.0 - np.cos(x[1:-1]-x[2:])*np.sin(x[1:-1]+x[2:]) + np.sin(x[1:-1]-x[2:])*np.cos(x[1:-1]+x[2:])
        
        # Jacobiano Tridiagonale
        J = diags([low_J, diag_J, up_J], [-1, 0, 1], shape=(n, n), format='csr')
        
        # Termine di primo ordine (Pentadiagonale)
        first_order = J.T @ J
        
        # Termine di secondo ordine (Diagonale dominante)
        diag_2 = 18.0 * x * f 
        second_order = diags(diag_2, 0, shape=(n, n), format='csr')

        return first_order + second_order