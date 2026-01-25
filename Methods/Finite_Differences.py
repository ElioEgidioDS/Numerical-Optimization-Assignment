import numpy as np
from scipy import sparse
from scipy import sparse

class FiniteDifferences:

    def __init__(self, problem_instance):
        self.problem = problem_instance

# builds the step of perturbation used to compute the approximation of the derivatives
    def step_vector(x, k, mode, x_ref = None, zero_floor = 1.0):
        
        eps = 10.0 ** (-k)

        # scalar mode: h = 10^{-k}
        if mode == "scalar":
            return np.full_like(x, eps, dtype=float)

        # adaptive mode: h_i = 10^{-k} * |x_ref_i|
        if mode == "adaptive":
            if x_ref is None:
                x_ref = x
            h = eps * np.abs(x_ref).astype(float)

            # safeguard floor to avoid zero steps when xi = 0 
            if zero_floor > 0.0:
                floor = eps * float(zero_floor)
                h = np.maximum(h, floor)

            return h

        raise ValueError("mode must be 'scalar' or 'adaptive'.")

# builds FD tridiagonal jacobian computing just the three non null diagonals
    def approximate_jacobian_tridiag_diagonals(
        self, x, k_step, step_mode, scheme = "centered", x_ref = None,
        zero_floor = 1.0):

        n = x.size
        h = self.step_vector(x, k=k_step, mode=step_mode, x_ref=x_ref, zero_floor=zero_floor)

        f0 = self.problem.function_k(x).ravel()
        if f0.shape[0] != n:
            raise ValueError("function_k(x) must return a residual vector of length n.")

        # d_main[i]  = J_{i, i}
        d_main = np.zeros(n, dtype=float)
        
        #d_lower[j] = J_{j+1, j}   for j = 0,...,n-2  (offset -1)
        d_lower = np.zeros(n - 1, dtype=float)

        # d_upper[i] = J_{i, i+1}   for i = 0,...,n-2  (offset +1)
        d_upper = np.zeros(n - 1, dtype=float)

        p = np.zeros(n, dtype=float)

        # iterate over non overlapping  sets
        for offset in (0, 1, 2):
            idx = np.arange(offset, n, 3)
            if idx.size == 0:
                continue
            
            # apply perturbation just to the current set
            p[idx] = h[idx]

            if scheme == "forward":
                f_plus = self.problem.function_k(x + p).ravel()
                diff = f_plus - f0
                scale = 1.0
            elif scheme == "centered":
                f_plus = self.problem.function_k(x + p).ravel()
                f_minus = self.problem.function_k(x - p).ravel()
                diff = f_plus - f_minus
                scale = 2.0
            else:
                raise ValueError("scheme must be 'forward' or 'centered'.")

            # reset perturbation value
            p[idx] = 0.0

            # main diagonal J_{j,j}
            d_main[idx] = diff[idx] / (scale * h[idx])

            # lower diagonal J_{j+1,j} stored at index j
            cols = idx[idx < n - 1]
            rows = cols + 1
            d_lower[cols] = diff[rows] / (scale * h[cols])

            # upper diagonal J_{j-1,j} stored at index (j-1)
            cols = idx[idx > 0]
            rows = cols - 1
            d_upper[rows] = diff[rows] / (scale * h[cols])

        return d_lower, d_main, d_upper

# recieves the three diagonals from the previous function and builds sparse Jacobian
    def approximate_jacobian_tridiag(self, x, k_step, step_mode, scheme="centered", x_ref=None, zero_floor=0.0, fmt="csr"):
        d_lower, d_main, d_upper = self.approximate_jacobian_tridiag_diagonals(
            x, k_step, step_mode, scheme, x_ref, zero_floor
        )
        return sparse.diags([d_lower, d_main, d_upper], offsets=[-1, 0, 1], shape=(x.size, x.size), format=fmt)


# approximates grad F(x) = J(x)^T f(x) in O(n), using just the three diagonals of the tridiagonal jacobian
    def approximate_gradient(self, x, k_step, step_mode, scheme = "centered", x_ref = None, zero_floor = 1.0):
        
        n = x.size
        f0 = self.problem.function_k(x).ravel()

        d_lower, d_main, d_upper = self.approximate_jacobian_tridiag_diagonals(
            x, k_step=k_step, step_mode=step_mode, scheme=scheme, x_ref=x_ref, zero_floor=zero_floor
        )

        g = np.zeros(n, dtype=float)

        # (J^T f)_i = d_main[i]*f[i] + d_upper[i-1]*f[i-1] + d_lower[i]*f[i+1]
        g += d_main * f0
        g[1:] += d_upper * f0[:-1]
        g[:-1] += d_lower[:] * f0[1:]

        return g

# approximates a pentadiagonal hessian by finite differences of the gradient:
# H_{i,j} ≈ (g_i(x + h_j e_j) - g_i(x)) / h_j
    def approximate_hessian_pentadiag(self, x, grad_fun, k_step, step_mode, x_ref = None,
        zero_floor = 1.0):

        n = x.size
        
        # computation of perturbation step
        h = self.step_vector(x, k=k_step, mode=step_mode, x_ref=x_ref, zero_floor=zero_floor)

        g0 = grad_fun(x).ravel()

        # initialization of diagonals
        diag0 = np.zeros(n, dtype=float)
        diag_p1 = np.zeros(n - 1, dtype=float)  # offset +1
        diag_p2 = np.zeros(n - 2, dtype=float)  # offset +2
        diag_m1 = np.zeros(n - 1, dtype=float)  # offset -1
        diag_m2 = np.zeros(n - 2, dtype=float)  # offset -2

        p = np.zeros(n, dtype=float)

        # iterate over 5 non overlapping sets
        for offset in range(5):
            idx = np.arange(offset, n, 5)
            if idx.size == 0:
                continue
            
            # apply perturbation to currect set
            p[idx] = h[idx]
            g1 = grad_fun(x + p).ravel()
            p[idx] = 0.0

            diff = g1 - g0

            # main diagonal H_{j,j}
            diag0[idx] = diff[idx] / h[idx]

            # upper diagonals: H_{j-1,j} -> offset +1 at position (j-1)
            j = idx[idx >= 1]
            if j.size > 0:
                diag_p1[j - 1] = diff[j - 1] / h[j]

            # H_{j-2,j} -> offset +2 at position (j-2)
            j = idx[idx >= 2]
            if j.size > 0:
                diag_p2[j - 2] = diff[j - 2] / h[j]

            # lower diagonals: H_{j+1,j} -> offset -1 at position j
            j = idx[idx <= n - 2]
            if j.size > 0:
                diag_m1[j] = diff[j + 1] / h[j]

            # H_{j+2,j} -> offset -2 at position j
            j = idx[idx <= n - 3]
            if j.size > 0:
                diag_m2[j] = diff[j + 2] / h[j]

        # population of sparse matrix
        H = sparse.diags(
            diagonals=[diag_m2, diag_m1, diag0, diag_p1, diag_p2],
            offsets=[-2, -1, 0, 1, 2],
            shape=(n, n),
            format="csr"
        )

        # forced symmetrization is crucial for (modified) newton methods
        H = (H + H.T) * 0.5
        return H

# helper: build a consistent gradient callable for the chosen FD settings
    def make_grad_fun(self, k_step, step_mode, scheme="centered", x_ref=None, zero_floor=1.0):
        def grad(y):
            return self.approximate_gradient(
                y,
                k_step=k_step,
                step_mode=step_mode,
                scheme=scheme,
                x_ref=x_ref,
                zero_floor=zero_floor
            )
        return grad