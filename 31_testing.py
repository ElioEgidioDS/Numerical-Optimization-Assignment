import numpy as np
import time
from Problems.Problem_31 import Problem_31
from Methods.NewtonMethod import NewtonMethod
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod

def main():
    # 1. Configurazione Parametri
    n = 100000
    tol = 1e-8
    max_iter = 1000
    
    # Inizializzazione Problema
    problem = Problem_31(n)
    x0 = problem.x0  # Il punto di partenza x_l = -1 definito nella classe
    
    print(f"--- Testing Problem 31 with n = {n} ---")
    print(f"Target Tolerance: {tol}")
    print("-" * 40)

    # 2. Test Newton Method (Esatto con Cholesky Banded)
    print("\n[Running Newton Method...]")
    # Parametri: tol, max_n, bck_trk_C1, bck_trk_phi
    nm = NewtonMethod(tol, max_iter, 1e-4, 0.5)
    
    start_nm = time.time()
    x_nm, path_nm, grad_norm_nm, conv_nm, k_nm = nm.minimize(problem, x0, mode="exact")
    end_nm = time.time() - start_nm
    
    if conv_nm:
        print(f"CONVERGED in {k_nm} iterations")
        print(f"Time: {end_nm:.4f} s")
        print(f"Final Gradient Norm: {grad_norm_nm:.2e}")
        print(f"Final Function Value: {problem.function(x_nm):.2e}")
    else:
        print("Newton Method failed to converge.")

    # 3. Test Truncated Newton Method
    print("\n" + "-" * 40)
    print("[Running Truncated Newton Method...]")
    # Parametri: tol, kmax (outer), jmax (inner), order_conv
    tn = TruncatedNewtonMethod(tol, max_iter, 500, 'sl')
    
    start_tn = time.time()
    x_tn, path_tn, grad_norm_tn, conv_tn, k_tn = tn.truncated_newton(
        problem.function, problem.gradient, problem.hessian, problem.x0)
    end_tn = time.time() - start_tn
    
    if conv_tn:
        print(f"CONVERGED in {k_tn} iterations")
        print(f"Time: {end_tn:.4f} s")
        print(f"Final Gradient Norm: {grad_norm_tn:.2e}")
        print(f"Final Function Value: {problem.function(x_tn):.2e}")
    else:
        print("Truncated Newton Method failed to converge.")

    # 4. Confronto finale
    print("\n" + "=" * 40)
    if conv_nm and conv_tn:
        diff = np.linalg.norm(x_nm - x_tn)
        print(f"Difference between NM and TN solutions: {diff:.2e}")

if __name__ == "__main__":
    main()