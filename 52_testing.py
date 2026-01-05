import numpy as np
import time
from Problems.Problem_52 import Problem_52
from Methods.ModifiedNewtonMethod import NewtonMethod
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from Methods.Finite_Differences import FiniteDifferences

class ProblemWrapper:
    """
    Wraps the Problem_52 instance to switch between Analytical and FD derivatives.
    """
    def __init__(self, prob, fd_handler, mode='analytical'):
        self.prob = prob
        self.fd = fd_handler
        self.mode = mode  # 'analytical', 'mixed_fd', 'full_fd'
        self.n = prob.n
        self.x0 = prob.x0

    def function(self, x):
        return self.prob.function(x)

    def gradient(self, x):
        if self.mode == 'full_fd':
            # Uses the FD Gradient (J^T * f)
            return self.fd.approximate_gradient(x, k_step=8, step_mode='adaptive', scheme='centered')
        return self.prob.gradient(x)

    def hessian(self, x):
        if self.mode == 'analytical':
            return self.prob.hessian(x)
        
        # Decide which gradient function to differentiate for the FD Hessian
        if self.mode == 'mixed_fd':
            # Differentiate the exact gradient
            g_fun = self.prob.gradient
        else:
            # Differentiate the FD gradient
            g_fun = lambda y: self.fd.approximate_gradient(y, k_step=8, step_mode='adaptive')
            
        return self.fd.approximate_hessian_pentadiag(x, grad_fun=g_fun, k_step=6, step_mode='adaptive')

def main():
    n = 10000  
    tol = 1e-6
    max_iter = 500
    
    # Initialization
    problem_base = Problem_52(n)
    fd_handler = FiniteDifferences(problem_base)
    
    # INITIALIZE METHODS WITH CORRECT ARGUMENTS
    # NewtonMethod(tol, max_n, bck_trk_rho, bck_trk_C1)
    nm = NewtonMethod(tol, max_iter, 0.5, 1e-4)
    
    # TruncatedNewtonMethod(tol, kmax, jmax, order_conv, rho, c1)
    tn = TruncatedNewtonMethod(tol, max_iter, 200, 'sl', 0.5, 1e-4)

    modes = ['analytical', 'mixed_fd', 'full_fd']
    
    print(f"--- Benchmark: Problem 52 | n={n} | x0=zeros ---")
    header = f"{'Method':<18} | {'Derivs':<12} | {'Status':<8} | {'Iter':<5} | {'Time (s)':<8} | {'Final F(x)':<10}"
    print(header)
    print("-" * len(header))

    for mode in modes:
        proxy = ProblemWrapper(problem_base, fd_handler, mode=mode)
        
        # 1. Newton Method (Modified / Banded)
        start = time.time()
        # Note: NewtonMethod uses 'minimize(problem, x0)'
        x_nm, _, g_norm_nm, conv_nm, iters_nm = nm.minimize(proxy, proxy.x0)
        t_nm = time.time() - start
        s_nm = "CONV" if conv_nm else "FAIL"
        f_nm = proxy.function(x_nm)
        print(f"{'Mod-Newton':<18} | {mode:<12} | {s_nm:<8} | {iters_nm:<5} | {t_nm:<8.4f} | {f_nm:<10.2e}")

        # 2. Truncated Newton Method (CG)
        start = time.time()
        # Note: TruncatedNewton uses 'truncated_newton(f, gradf, hessf, x0)'
        x_tn, g_norm_tn, conv_tn, iters_tn, _ = tn.truncated_newton(
            proxy.function, proxy.gradient, proxy.hessian, proxy.x0)
        t_tn = time.time() - start
        s_tn = "CONV" if conv_tn else "FAIL"
        f_tn = proxy.function(x_tn)
        print(f"{'Trunc-Newton':<18} | {mode:<12} | {s_tn:<8} | {iters_tn:<5} | {t_tn:<8.4f} | {f_tn:<10.2e}")
        print("-" * len(header))

if __name__ == "__main__":
    main()