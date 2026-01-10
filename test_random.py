import numpy as np
import time
import pandas as pd

# Import your specific classes
from Problems.Problem_31 import Problem_31
from Problems.Problem_52 import Problem_52
from Problems.Problem_fd import Problem_fd
from Methods.ModifiedNewtonMethod import ModifiedNewtonMethod
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from Methods.Finite_Differences import FiniteDifferences

def main():
    # --- Configuration ---
    n_dim = 10000
    tol = 1e-6
    max_iter = 1000
    
    # Set seed for reproducibility
    np.random.seed(42)

    # --- Initialize Solvers ---
    # Using parameters consistent with your files
    nm_solver = ModifiedNewtonMethod(tol=tol, max_n=max_iter, bck_trk_rho=0.5, bck_trk_C1=1e-4)
    tn_solver = TruncatedNewtonMethod(tol=tol, kmax=max_iter, jmax=500, order_conv='sl', rho=0.5, c1=1e-4)

    print(f"Running Full Test Suite")
    print(f"Dim: {n_dim} | Tol: {tol} | MaxIter: {max_iter}")
    print("=" * 130)
    print(f"{'Prob':<5} | {'Point':<12} | {'Method':<12} | {'Mode':<10} | {'Status':<10} | {'Reason':<10} | {'Iter':<5} | {'Time':<8} | {'Grad Norm'}")
    print("-" * 130)

    # List of base problems
    base_problems = [
        (Problem_31(n_dim), "P31"),
        (Problem_52(n_dim), "P52")
    ]

    for prob_instance, prob_name in base_problems:
        
        # --- 1. Setup Derivative Modes ---
        fd = FiniteDifferences(prob_instance)
        
        # A. Exact
        p_exact = prob_instance
        
        # B. Mixed (Exact Grad, FD Hessian)
        def hess_mixed_func(x):
            return fd.approximate_hessian_pentadiag(
                x, 
                grad_fun=prob_instance.gradient, 
                k_step=8, 
                step_mode="adaptive", 
                zero_floor=1e-2
            )
        p_mixed = Problem_fd(prob_instance, prob_instance.gradient, hess_mixed_func)
        
        # C. Full FD (FD Grad, FD Hessian)
        def grad_fd_func(x):
            return fd.approximate_gradient(x, k_step=8, step_mode="adaptive", zero_floor=1e-2)
            
        def hess_fd_func(x):
            return fd.approximate_hessian_pentadiag(
                x, 
                grad_fun=grad_fd_func, 
                k_step=8, 
                step_mode="adaptive", 
                zero_floor=1e-2
            )
        p_fd = Problem_fd(prob_instance, grad_fd_func, hess_fd_func)

        modes = [("Exact", p_exact), ("Mixed", p_mixed), ("Full FD", p_fd)]

        # --- 2. Setup Starting Points ---
        # Suggested Point
        x_suggested = prob_instance.x0.copy()
        
        # Random Point (Suggested +/- 1.0)
        # Note: We generate a FRESH random perturbation for each problem
        perturbation = np.random.uniform(-1, 1, size=n_dim)
        x_random = x_suggested + perturbation

        starting_points = [
            ("Suggested", x_suggested),
            ("Random", x_random)
        ]

        # --- 3. Run Tests ---
        for pt_name, x_start in starting_points:
            for mode_name, current_prob in modes:
                
                # We copy x_start to ensure methods don't modify the source for the next run
                
                # --- Method 1: Modified Newton ---
                start = time.time()
                try:
                    # Returns: x, grad_norm, converge, reason, k, path
                    _, g_norm, conv, reason, k, _ = nm_solver.modified_newton(current_prob, x_start.copy())
                    status = "CONV" if conv else "FAIL"
                except Exception as e:
                    status = "ERR"
                    reason = "Crash"
                    k = 0
                    g_norm = 0.0
                elapsed = time.time() - start
                
                print(f"{prob_name:<5} | {pt_name:<12} | {'ModNewton':<12} | {mode_name:<10} | {status:<10} | {reason:<10} | {k:<5} | {elapsed:<8.2f} | {g_norm:.2e}")

                # --- Method 2: Truncated Newton ---
                start = time.time()
                try:
                    # Returns: x, grad_norm, converge, reason, k, path
                    _, g_norm, conv, reason, k, _ = tn_solver.truncated_newton(
                        current_prob.function, 
                        current_prob.gradient, 
                        current_prob.hessian, 
                        x_start.copy()
                    )
                    status = "CONV" if conv else "FAIL"
                except Exception as e:
                    status = "ERR"
                    reason = "Crash"
                    k = 0
                    g_norm = 0.0
                elapsed = time.time() - start

                print(f"{prob_name:<5} | {pt_name:<12} | {'TruncNewt':<12} | {mode_name:<10} | {status:<10} | {reason:<10} | {k:<5} | {elapsed:<8.2f} | {g_norm:.2e}")

        print("-" * 130)

if __name__ == "__main__":
    main()