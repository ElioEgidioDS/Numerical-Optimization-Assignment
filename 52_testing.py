import numpy as np
import time
from Problems.Problem_52 import Problem_52
from Methods.NewtonMethod import NewtonMethod
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod

def main():
    # 1. Configurazione Parametri
    n = 10000          # Dimensione del problema
    tol = 1e-6       # Tolleranza (meno stringente per via della natura oscillatoria)
    max_iter = 1000    # Massimo numero di iterazioni esterne
    
    # Inizializzazione Problema 52
    problem = Problem_52(n)
    
    # --- NUOVO PUNTO DI PARTENZA RANDOM ---
    # Usiamo un seed per rendere il test confrontabile in futuro
    # Generiamo valori tra -0.5 e 0.5 per evitare stalli immediati in zone troppo piatte
    x0 = np.random.uniform(-0.5, 0.5, n) 
    # --------------------------------------
    #x0 = problem.x0
    print(f"--- Testing Problem 52 (Trigexp 1) with n = {n} (RANDOM START) ---")
    print(f"Target Tolerance: {tol}")
    print("-" * 40)

    # 2. Test Newton Method (Esatto con Cholesky Banded)
    print("\n[Running Newton Method (Exact Banded)...]")
    # Parametri: tol, max_n, bck_trk_C1, bck_trk_phi
    nm = NewtonMethod(tol, max_iter, 1e-4, 0.5)
    
    start_nm = time.time()
    # Il metodo 'exact' sfrutta la struttura pentadiagonale dell'Hessiana
    x_nm, path_nm, grad_norm_nm, conv_nm, k_nm = nm.minimize(problem, x0, mode="exact")
    end_nm = time.time() - start_nm
    
    if conv_nm:
        print(f"CONVERGED in {k_nm} iterations")
        print(f"Time: {end_nm:.4f} s")
        print(f"Final Gradient Norm: {grad_norm_nm:.2e}")
        print(f"Final Function Value: {problem.function(x_nm):.2e}")
    else:
        print("Newton Method failed to converge.")

    # 3. Test Truncated Newton Method (Iterativo con CG)
    print("\n" + "-" * 40)
    print("[Running Truncated Newton Method (CG)...]")
    # Parametri: tol, kmax (outer), jmax (inner), order_conv
    # jmax=200 per gestire meglio le oscillazioni trigonometriche
    tn = TruncatedNewtonMethod(tol, max_iter, 200, 'sl')
    
    start_tn = time.time()
    x_tn, path_tn, grad_norm_tn, conv_tn, k_tn = tn.truncated_newton(
        problem.function, problem.gradient, problem.hessian, x0)
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
        print(f"Difference between NM and TN solutions (L2-norm): {diff:.2e}")
        print(f"Performance Ratio: TN is {end_nm/end_tn:.2f}x faster/slower than NM")

if __name__ == "__main__":
    main()