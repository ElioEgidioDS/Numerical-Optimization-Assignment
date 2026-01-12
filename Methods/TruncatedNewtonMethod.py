import numpy as np
from scipy.sparse.linalg import spilu, spsolve_triangular
from scipy.sparse import issparse, csr_matrix, csc_matrix
from Methods.Backtracking import Backtracking



class TruncatedNewtonMethod:

    def __init__(self, tol, kmax, jmax, order_conv='sl', rho=0.5, c1=1e-4):
        self.tol = tol       
        self.kmax = kmax    
        self.jmax = jmax      
        self.order_conv = order_conv
        self.rho = rho
        self.c1 = c1
    

    def forcing_term(self, gradient_norm):
        if (self.order_conv == 'l'):
            return 0.5
        elif (self.order_conv == 'sl'):
            return min(0.5, np.sqrt(gradient_norm))
        else:
            return min(0.5, gradient_norm)
            
    

    # # computed product B_precond @ d
    # # L, U: result of factorization

    # def mat_vec_precond(self, B, d, L, U): 
    #     # full product to be computed: B_preconditioned @ d 
    #     # equivalent to L^{-1} @ B @ U^{-1} @ d
    #     # divided into three sub operations

    #     # [U^{-1} @ d] found by solving linear system [U @ prod1 = d]
    #     prod1 = spsolve_triangular(U, d, lower=False)
        
    #     # [B @ prod1] computed directly
    #     prod2 = B @ prod1

    #     # [L^{-1} @ prod2] found by solving linear system [L @ prod3 = prod2]
    #     prod3 = spsolve_triangular(L, prod2, lower=True)
        
    #     return prod3




    # solves the inner linear system Bz = c to find the descent direction pk_{tn} 
        # B = hessian(f)
        # c = - grad(f)
        # z = p
    def inner_CG(self, B, c, z0, etak):

        failure_reason = "-"
        # Initialization, start with p = 0 (so xk = 0 in local space)
        p_sol = np.zeros_like(c) 
        
        # r = c - B*p_sol = c
        r = c.copy()
        
        # d is the search direction
        d = r.copy()
        
        # residual norm for stopping condition
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-16: 
            return p_sol, failure_reason

        # CG Loop
        for j in range(self.jmax):
            
            # MATRIX-VECTOR PRODUCT 
            # this is the only expensive step
            Bd = B @ d
            
            # curvature check (d^T B d) ---
            dBd = np.dot(d, Bd)
            

            # If we encounter negative curvature (indefinite Hessian),
            # we must stop and return the best direction found so far.
            if dBd <= 1e-12:
                if j == 0:
                    # if it happens at the very first step, 
                    # the Hessian is bad immediately. Return steepest descent (c).
                    return c, failure_reason
                else:
                    # otherwise, return the accumulated solution
                    return p_sol, failure_reason
            
            # standard CG
            alpha = np.dot(r, r) / dBd
            
            p_next = p_sol + alpha * d
            r_next = r - alpha * Bd
            
            # we check convergence using a relative residual)
            if np.linalg.norm(r_next) / norm_c < etak:
                return p_next, failure_reason
            
            # update search direction for next step
            beta = np.dot(r_next, r_next) / np.dot(r, r)
            d = r_next + beta * d
            
            # update variables
            r = r_next
            p_sol = p_next
            
        failure_reason = "GC"
        return p_sol, failure_reason


    def truncated_newton(self, f, gradf, hessf, x0):
        
        xk = x0.copy()
        xk_sequence = [xk.copy()]
        grad_xk = gradf(xk)
        grad_xk_norm = np.linalg.norm(grad_xk)
        flag_convergence = False
        failure_reason = "-"
        bcktrk = Backtracking(self.c1,self.rho,100)

        for k in range(self.kmax): 
            
            if grad_xk_norm < self.tol:
                flag_convergence = True
                return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence), "-"
            
            if not np.isfinite(grad_xk_norm):
                return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence), "NaN"
            
            eta_k = self.forcing_term(grad_xk_norm)

            B = hessf(xk)
            c = -grad_xk
            z0 = np.zeros_like(c)    
            
            # inner conjugate gradient method to solve linear system and find p_tn
            p_tn, failure_CG = self.inner_CG(B, c, z0, eta_k)

            if np.dot(p_tn, grad_xk) >= 0:
                p_tn = c

            alpha = bcktrk.backtrack(p_tn,xk,f,1,grad_xk)
            if alpha == 0:
                 return xk, grad_xk_norm, False, k, np.array(xk_sequence), "LS"
            xk = xk + alpha * p_tn   
            xk_sequence.append(xk.copy())

            grad_xk = gradf(xk)
            grad_xk_norm = np.linalg.norm(grad_xk)

        print("TRN DID NOT CONVERGE")
        print("final k: ",k)
        print("final alpha: ", alpha)
        print("final norm of the gradient: ",grad_xk_norm)

        if failure_CG != "-":
            failure_reason = failure_CG
        else:
            failure_reason = "MAX"
            return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence), failure_reason
