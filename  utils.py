import numpy as np
from scipy.linalg import spilu, spsolve_triangular
from scipy.sparse import issparse, csr_matrix, csc_matrix

def line_search(f, gradf, xk, p, alpha=1, rho=0.5, c1=1e-4):
    fxk = f(xk) #reduces function evaluations in loop
    grad_fxk = gradf(xk)
    while (f(xk + alpha * p) > fxk + (c1 * alpha * (grad_fxk @ p))):
        alpha *= rho
    return alpha

def forcing_term(gradient_norm, order_conv):
    if (order_conv == 'l'):
        return 0.5
    elif (order_conv == 'sl'):
        return min(0.5, np.sqrt(gradient_norm))
    else:
        return min(0.5, gradient_norm)
        

# B: matrix 
# d: vector
# L: preconditioner (to be applied both on right and left)

def mat_vec_precond(B, d, L, U): 
    # full product to be computed: B_preconditioned @ d 
    # equivalent to L^{-1} @ B @ U^{-1} @ d
    # divided into three sub operations, using solve_banded (keeping sparsity)

    # [U^{-1} @ d] found by solving linear system [U @ prod1 = d]
    prod1 = spsolve_triangular(L, d, lower=True)
    
    # [B @ prod1] computed directly
    prod2 = B @ prod1

    # [L^{-1} @ prod2] found by solving linear system [L @ prod3 = prod2]
    prod3 = spsolve_triangular(U, prod2, lower=False)
    
    return prod3




# solves the inner linear system Bz = c to find the descent direction pk_{tn} 
    # B = hessian(f)
    # c = - grad(f)
    # z = p

def inner_CG(B, c, z0, jmax, tol):

    if not issparse(B):
        B = csr_matrix(B)
    B = B.tocsc()
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


    while (j < jmax and relres > tol): 
        
        # B @ dk   ==>   B_prime @ dk 
        Bd_cond = mat_vec_precond(B, dk, L, U)

        # dk.T @ B @ dk    ==>   dk @ B_prime @ dk (not explicitely)   
        dBd_cond = dk.T @ Bd_cond
        
        # check if postitive definite
        # theoretically dBd > 0, but bc of machine precision we use a tolerance
        if (dBd_cond > 1e-12):      
                    
            alpha = (rk.T @ rk) / dBd_cond
            y = y + alpha * dk
            r_next = rk - alpha * Bd_cond
            beta = (r_next.T @ r_next)/(rk.T @ rk)
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




def truncated_newton(f, gradf, hessf, x0, tolgrad, kmax, jmax, order_conv='sl'):
    
    xk = x0.copy()
    grad_xk = gradf(xk)
    grad_xk_norm = np.linalg.norm(grad_xk)
    k = 0

    while (k < kmax and grad_xk_norm >= tolgrad):
        
        k += 1
        eta_k = forcing_term(grad_xk_norm, order_conv)

        B = hessf(xk)
        c = -grad_xk
        z0 = np.zeros_like(c)    
        
        # inner conjugate gradient method to solve linear system and find p_tn
        p_tn = inner_CG(B, c, z0, jmax, eta_k)

        alpha = line_search(f, gradf, xk, p_tn, alpha=1, rho=0.5, c1=1e-4)
        
        xk = xk + alpha * p_tn   

        grad_xk = gradf(xk)
        grad_xk_norm = np.linalg.norm(grad_xk)
   
    return xk



def convert_to_banded(sparse_mat):

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