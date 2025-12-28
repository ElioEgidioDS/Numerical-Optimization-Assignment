import numpy as np
from scipy.sparse.linalg import spilu, spsolve_triangular
from scipy.sparse import issparse, csr_matrix, csc_matrix


class TruncatedNewtonMethod:

    def __init__(self, tol, kmax, jmax, order_conv='sl'):
        self.tol = tol       
        self.kmax = kmax    
        self.jmax = jmax      
        self.order_conv = order_conv


    def line_search(self, f, gradf, xk, p, alpha=1, rho=0.5, c1=1e-4):
        fxk = f(xk)
        grad_fxk = gradf(xk)
        
        # Calcolo pendenza lungo la direzione p
        slope = np.dot(grad_fxk, p) 
        if slope >= 0: return 0

        while (f(xk + alpha * p) > fxk + (c1 * alpha * slope)):
            alpha *= rho
            if alpha < 1e-12: break # to avoid inf loops
        return alpha

    def forcing_term(self, gradient_norm):
        if (self.order_conv == 'l'):
            return 0.5
        elif (self.order_conv == 'sl'):
            return min(0.5, np.sqrt(gradient_norm))
        else:
            return min(0.5, gradient_norm)
            
    

    # computed product B_precond @ d
    # L, U: result of factorization
    def mat_vec_precond(self, B, d, L, U): 
        # full product to be computed: B_preconditioned @ d 
        # equivalent to L^{-1} @ B @ U^{-1} @ d
        # divided into three sub operations

        # [U^{-1} @ d] found by solving linear system [U @ prod1 = d]
        prod1 = spsolve_triangular(U, d, lower=False)
        
        # [B @ prod1] computed directly
        prod2 = B @ prod1

        # [L^{-1} @ prod2] found by solving linear system [L @ prod3 = prod2]
        prod3 = spsolve_triangular(L, prod2, lower=True)
        
        return prod3




    # solves the inner linear system Bz = c to find the descent direction pk_{tn} 
        # B = hessian(f)
        # c = - grad(f)
        # z = p
    def inner_CG(self, B, c, z0, etak):

        if not issparse(B):
            B = csr_matrix(B)

    # Incomplete LU factorization B ~ L * U 
        try: 
            ilu_fact = spilu(B, fill_factor=10, drop_tol=1e-4)
            L = ilu_fact.L
            U = ilu_fact.U
        except (RuntimeError, ValueError):
            return c.copy()
        
        # B'y = c'    
        # c' = L^{-1} * c by solving L^T c = c'
        c_prime = spsolve_triangular(L, c, lower=True)
        
        y = np.zeros_like(z0)

        # initializations
        rk = c_prime.copy()
        dk = rk.copy()
        
        norm_c_prime = np.linalg.norm(c_prime)
        if norm_c_prime < 1e-16: 
            #it would be considered as zero and the division would fail
            return np.zeros_like(z0)
        
        relres = np.linalg.norm(rk) / norm_c_prime
        j = 0


        while (j < self.jmax and relres > etak): 
            
            # B @ dk   ==>   B_prime @ dk 
            Bd_cond = self.mat_vec_precond(B, dk, L, U)

            # dk.T @ B @ dk    ==>   dk @ B_prime @ dk (not explicitely)   
            dBd_cond = np.dot(dk.T, Bd_cond)
            
            # check if postitive definite
            # theoretically dBd > 0, but bc of machine precision we use a tolerance
            if (dBd_cond > 1e-12):      
                        
                alpha = np.dot(rk.T, rk) / dBd_cond
                y = y + alpha * dk
                r_next = rk - alpha * Bd_cond
                beta = np.dot(r_next.T, r_next)/np.dot(rk.T, rk)
                dk = r_next + beta * dk
                rk = r_next
                
                relres = np.linalg.norm(rk) / norm_c_prime
                j += 1

            else: # not positive definite
                
                if j == 0: # if it stopped at the first iteration
                    return spsolve_triangular(U, dk, lower=False)
                
                else: # if it stopped ater the first iteration
                    return spsolve_triangular(U, y, lower=False)
                    
        return spsolve_triangular(U, y, lower=False)


    def truncated_newton(self, f, gradf, hessf, x0):
        
        xk = x0.copy()
        xk_sequence = [xk.copy()]
        grad_xk = gradf(xk)
        grad_xk_norm = np.linalg.norm(grad_xk)
        k = 0
        flag_convergence = False

        for k in range(self.kmax): 
            
            if grad_xk_norm < self.tol:
                flag_convergence = True
                return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence)
            
            eta_k = self.forcing_term(grad_xk_norm)

            B = hessf(xk)
            c = -grad_xk
            z0 = np.zeros_like(c)    
            
            # inner conjugate gradient method to solve linear system and find p_tn
            p_tn = self.inner_CG(B, c, z0, eta_k)

            if np.dot(p_tn, grad_xk) >= 0:
                p_tn = c

            alpha = self.line_search(f, gradf, xk, p_tn, alpha=1, rho=0.5, c1=1e-4)
            
            xk = xk + alpha * p_tn   
            xk_sequence.append(xk.copy())

            grad_xk = gradf(xk)
            grad_xk_norm = np.linalg.norm(grad_xk)

        print("TRN DID NOT CONVERGE")
        print("final k: ",k)
        print("final alpha: ", alpha)
        print("final norm of the gradient: ",grad_xk_norm)

        return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence)