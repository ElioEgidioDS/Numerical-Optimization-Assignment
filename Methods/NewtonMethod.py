import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import eye
from Methods.Backtracking import Backtracking

class NewtonMethod:
    def __init__(self, tol, max_n):
        self.tol = tol
        self.max_n = max_n

    def minimize(self, problem,x0, mode, k=0):
        bcktrk = Backtracking(0.5,0.5,100)
        x = x0
        B = 1e-3
        if mode == "exact":
            for k in range(self.max_n):
                gradient = problem.gradient(x)
                hessian = problem.hessian(x)

                if np.linalg.norm(gradient) < self.tol*max(1,np.linalg.norm(gradient)):
                    return x
                
                if hessian.diagonal().min() > 0:
                    tau = 0
                else:
                    tau = B - hessian.diagonal().min()
                
                for j in range(20):
                    try:
                        Bk = hessian + tau*eye(self.max_n,format = "csc")
                        lu_fact = splu(Bk)

                        p_mn = lu_fact.solve(-gradient)
                    except RuntimeError:
                        tau = max(2 * tau, B)

                alpha = bcktrk.backtrack(p_mn,x,problem.function,1,gradient)

                x = x + alpha * p_mn

            return x


        elif mode == "fd":
            ...#use approximated gradient