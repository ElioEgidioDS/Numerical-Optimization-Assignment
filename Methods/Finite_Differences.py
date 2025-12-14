import numpy as np
from scipy import sparse

class FiniteDifferences:    
    def __init__(self, problem_instance):
        """
        Initializes the instance with a problem object.
        The 'problem_instance' object MUST have a .function_k(x) method
        that returns the residual vector (for Least Squares problems).
        """
        self.problem = problem_instance
        

    # 1. STEP SIZE MANAGEMENT
    @staticmethod
    def calculate_step(x, k, mode='adaptive'):
        """
        Calculates the optimal perturbation step h: h = epsilon * max(|x|, 1.0).
        k: exponent for epsilon (e.g., k=6 -> epsilon=1e-6).
        """
        epsilon = 10.0**(-k)
        
        if mode == 'scalar':
            return np.full_like(x, epsilon, dtype=float)
        
        elif mode == 'adaptive':
            threshold = 1.0  
            magnitude = np.abs(x)
            safe_magnitude = np.where(magnitude < threshold, threshold, magnitude)
            return epsilon * safe_magnitude
        
        else:
            raise ValueError("Mode not supported. Use 'scalar' or 'adaptive'.")


    # 2. APPROXIMATE GRADIENT
    def approximate_gradient(self, x, k_step=6, scheme='centered'):
        """
        Calculates the gradient for F(x) = 0.5 * ||f(x)||^2 exploiting tridiagonal 
        sparsity (simulating A.T @ f(x) where J_ij = dA_i/dx_j).
        Computational Cost: O(N) instead of O(N^2).
        """
        n = x.size
        h_vec = self.calculate_step(x, k=k_step, mode='adaptive')
        
        f0 = self.problem.function_k(x)
        if f0.ndim == 0:
            raise ValueError("The function_k method must return a residual vector.")

        d_main = np.zeros(n)
        d_upper = np.zeros(n-1)
        d_lower = np.zeros(n-1)
        p = np.zeros(n)
        
        for offset in [0, 1, 2]:
            indices = np.arange(offset, n, 3)
            if indices.size == 0: continue
            
            p[indices] = h_vec[indices]
            
            # Calculate finite differences
            if scheme == 'forward':
                f_plus = self.problem.function_k(x + p)
                diff_vec = (f_plus - f0)
                step = h_vec
            elif scheme == 'centered':
                f_plus = self.problem.function_k(x + p)
                f_minus = self.problem.function_k(x - p)
                diff_vec = (f_plus - f_minus) / 2.0
                step = h_vec 
            
            p[indices] = 0.0 # Reset perturbation
                
            # Extract Main Diagonal (J_ii)
            d_main[indices] = diff_vec[indices] / step[indices]
            
            # Extract Lower Diagonal (J_{i+1, i})
            # Perturbation on column 'j' affects row 'j+1'
            cols_lower = indices[indices < n-1]
            rows_lower = cols_lower + 1
            d_lower[cols_lower] = diff_vec[rows_lower] / step[cols_lower]
            
            # Extract Upper Diagonal (J_{i-1, i})
            # Perturbation on column 'j' affects row 'j-1'
            cols_upper = indices[indices > 0]
            rows_upper = cols_upper - 1
            d_upper[rows_upper] = diff_vec[rows_upper] / step[cols_upper]

        # Construct sparse Jacobian
        J = sparse.diags([d_lower, d_main, d_upper], [-1, 0, 1], shape=(n, n))
        
        # Gradient: J.T @ f(x)
        grad = J.T @ f0
        
        return grad


    # 3. TRIDIAGONAL HESSIAN

    def finite_differences_H(self, x, gradient_function=None, k_step=5):
        """
        Calculates the Hessian assuming a tridiagonal structure using 3 gradient 
        evaluations.
        
        Args:
            x: evaluation point.
            gradient_function: (Optional) Function that calculates the gradient. 
                               If None, uses self.approximate_gradient.
            k_step: precision step for h.
        """
        n = x.shape[0]
        h = self.calculate_step(x, k=k_step, mode='adaptive')
        
        if gradient_function is None:
            # Default: Centered Approximate Gradient
            grad_handle = lambda y: self.approximate_gradient(y, k_step=6, scheme='centered')
        else:
            grad_handle = gradient_function

        # 1. Base Gradient
        g0 = grad_handle(x)
        
        diag_main = np.zeros(n)
        diag_upper = np.zeros(n-1)
        diag_lower = np.zeros(n-1)
        p = np.zeros(n)
        
        for offset in [0, 1, 2]:
            indices = np.arange(offset, n, 3)
            if len(indices) == 0: continue
                
            p[indices] = h[indices]
            g_perturbed = grad_handle(x + p)
            p[indices] = 0.0 # Reset
            
            diff_vec = (g_perturbed - g0)
            
            # 1. Main Diagonal (i, i)
            diag_main[indices] = diff_vec[indices] / h[indices]
            
            # 2. Upper Diagonal (i, i+1)
            # Element (i, i+1) of the Hessian matrix.
            # Obtained by perturbing column i+1 (indices) and reading row i (rows_affected)
            rows_affected_upper = indices - 1
            valid_mask_upper = rows_affected_upper >= 0
            
            valid_rows = rows_affected_upper[valid_mask_upper] # Rows (i)
            valid_cols = indices[valid_mask_upper]             # Perturbed Columns (i+1)
            
            # In sparse.diags, diagonal '1' (upper), element k corresponds to (k, k+1)
            # So we use valid_rows as index for the diagonal vector
            diag_upper[valid_rows] = diff_vec[valid_rows] / h[valid_cols]
            
            # 3. Lower Diagonal (i, i-1)
            rows_affected_lower = indices + 1
            valid_mask_lower = rows_affected_lower < n
            
            valid_rows = rows_affected_lower[valid_mask_lower]
            valid_cols = indices[valid_mask_lower]
            
            # In sparse.diags, diagonal '-1' (lower), element k corresponds to (k+1, k)
            # So we use valid_cols as index for the diagonal vector
            diag_lower[valid_cols] = diff_vec[valid_rows] / h[valid_cols]

        H_approx = sparse.diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], shape=(n, n), format='csc')
        
        # Symmetrization for numerical stability
        H_sym = (H_approx + H_approx.T) / 2
        
        return H_sym