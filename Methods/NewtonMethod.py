import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import eye
from Methods.Backtracking import Backtracking
from scipy.linalg import cholesky_banded, solve_banded



class NewtonMethod:
    def __init__(self, tol, max_n):
        self.tol = tol
        self.max_n = max_n

    def minimize(self, problem,x0, mode, k=0):
        bcktrk = Backtracking(0.5,0.5,100)
        x = x0.copy()
        path = [x.copy()]
        B = 1e-3
        I = eye(x0.shape[0], format="csr")

        if mode == "exact":
            for _ in range(self.max_n):
                gradient = problem.gradient(x)
                hessian = problem.hessian(x)

                if np.linalg.norm(gradient) < self.tol: #*max(1,np.linalg.norm(gradient))
                    return x, np.array(path), np.linalg.norm(gradient)
                
                if hessian.diagonal().min() > 0:
                    tau = 0
                else:
                    tau = B - hessian.diagonal().min()
                
                for j in range(20):
                    try:
                        Bk = hessian + tau*I
                        lu_fact = splu(Bk)#to change

                        p_mn = lu_fact.solve(-gradient)
                        break
                    except RuntimeError:
                        tau = max(2 * tau, B)

                alpha = bcktrk.backtrack(p_mn,x,problem.function,1,gradient)
                

                x = x + alpha * p_mn
                path.append(x.copy())
            print(alpha)
            print(np.linalg.norm(gradient))

            return x, np.array(path), np.linalg.norm(gradient)


        elif mode == "fd":
            ...#use approximated gradient