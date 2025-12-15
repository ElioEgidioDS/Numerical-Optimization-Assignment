import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import eye
from Methods.Backtracking import Backtracking
from scipy.linalg import cholesky_banded, cho_solve_banded, LinAlgError



class NewtonMethod:
    def __init__(self, tol, max_n):
        self.tol = tol
        self.max_n = max_n

    def convert_to_banded(self, sparse_mat):

        # convert form sparse to dense matrix representation
        n = sparse_mat.shape[0]
        banded_format = np.zeros((3, n))
        dense_mat = sparse_mat.tocoo()

        # extract diagonals needed containing non zero entries
        # we extract just the lower (or upper) ones, since the matix is symmetric
        diag = dense_mat.diagonal(k=0)
        first_lower = dense_mat.diagonal(k=-1)
        second_lower = dense_mat.diagonal(k=-2)

        #populate banded representation
        banded_format[0, 2:] = second_lower #dim: n-2
        banded_format[1, 1:] = first_lower #dim: n-1
        banded_format[2, :] = diag #dim: n

        return banded_format


    def minimize(self, problem,x0, mode, k=0):
        bcktrk = Backtracking(0.5,0.5,100)
        x = x0.copy()
        path = [x.copy()]
        B = 1e-3
        I = eye(x0.shape[0], format="csr")
        converge = False

        if mode == "exact":
            
            for k in range(self.max_n):
                gradient = problem.gradient(x)
                hessian = problem.hessian(x)

                if np.linalg.norm(gradient) < self.tol: #*max(1,np.linalg.norm(gradient)) #CHECK FOR OTHER STOPPING CRITERIONS
                    converge = True
                    return x, np.array(path), np.linalg.norm(gradient), converge, k
                
                if hessian.diagonal().min() > 0:
                    tau = 0
                else:
                    tau = B - hessian.diagonal().min()
                
                for j in range(20):
                    try:
                        Bk = hessian + tau*I
                        B_k_banded = self.convert_to_banded(Bk)
                        R = cholesky_banded(B_k_banded)
                    

                        #p_mn = lu_fact.solve(-gradient)
                        
                        break
                    except LinAlgError:
                        tau = max(2 * tau, B)
               
                p_mn = cho_solve_banded((R,False), -gradient)
                alpha = bcktrk.backtrack(p_mn,x,problem.function,1,gradient)
                
                #print(f"{k}:{np.linalg.norm(gradient)}")
                x = x + alpha * p_mn
                path.append(x.copy())

            print("DID NOT CONVERGE")
            print("final k: ",k)
            print("final alpha: ", alpha)
            print("final norm of the gradient: ",np.linalg.norm(gradient))

            return x, np.array(path), np.linalg.norm(gradient), converge, self.max_n


        elif mode == "fd":
            ...#use approximated gradient