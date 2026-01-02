import numpy as np
import time
from Problems.Problem_31 import Problem_31
from Methods.NewtonMethod import NewtonMethod
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from Methods.Finite_Differences import FiniteDifferences

class ProblemWrapper:
    def __init__(self, prob, fd_handler, mode='analytical'):
        self.prob = prob
        self.fd = fd_handler
        self.mode = mode
        self.n = prob.n

    def function(self, x): return self.prob.function(x)

    def gradient(self, x):
        if self.mode == 'full_fd':
            return self.fd.approximate_gradient(x, k_step=8, step_mode='adaptive', scheme='centered', zero_floor=1.0)
        return self.prob.gradient(x)

    def hessian(self, x):
        if self.mode == 'analytical': return self.prob.hessian(x)
        g_fun = self.prob.gradient if self.mode == 'mixed_fd' else lambda y: self.gradient(y)
        return self.fd.approximate_hessian_pentadiag(x, grad_fun=g_fun, k_step=6, step_mode='adaptive', zero_floor=1.0)

def main():
    n = 10000
    tol = 1e-6
    max_iter = 500
    
    problem_base = Problem_31(n)
    fd_handler = FiniteDifferences(problem_base)
    
    nm = NewtonMethod(tol, max_iter, 0.5, 1e-4)
    tn = TruncatedNewtonMethod(tol, max_iter, 200, 'sl', 0.5, 1e-4)

    # DEFINIZIONE PUNTI DI PARTENZA
    starting_points = {
        "PDF Standard (x0 = -1)": np.full(n, -1.0),
        "Random [-2, 2]": np.random.uniform(-1.5, -0.5, n)
    }

    modes = ['analytical', 'mixed_fd', 'full_fd']
    
    for sp_name, x0 in starting_points.items():
        print(f"\n{'='*85}")
        print(f" TEST RUN: {sp_name} | n={n}")
        print(f"{'='*85}")
        header = f"{'Method':<15} | {'Derivs':<12} | {'Status':<8} | {'Iter':<5} | {'Time (s)':<8} | {'Final F(x)':<10}"
        print(header)
        print("-" * len(header))

        for mode in modes:
            proxy = ProblemWrapper(problem_base, fd_handler, mode=mode)
            
            # 1. Mod-Newton
            start = time.time()
            x_nm, _, g_nm, conv_nm, iters_nm = nm.minimize(proxy, x0)
            t_nm = time.time() - start
            print(f"{'Mod-Newton':<15} | {mode:<12} | {'CONV' if conv_nm else 'FAIL':<8} | {iters_nm:<5} | {t_nm:<8.4f} | {proxy.function(x_nm):<10.2e}")

            # 2. Trunc-Newton
            start = time.time()
            x_tn, g_tn, conv_tn, iters_tn, _ = tn.truncated_newton(proxy.function, proxy.gradient, proxy.hessian, x0)
            t_tn = time.time() - start
            print(f"{'Trunc-Newton':<15} | {mode:<12} | {'CONV' if conv_tn else 'FAIL':<8} | {iters_tn:<5} | {t_tn:<8.4f} | {proxy.function(x_tn):<10.2e}")
            print("-" * len(header))

if __name__ == "__main__":
    main()