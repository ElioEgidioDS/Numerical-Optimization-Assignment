# %% [markdown]
# # Truncated Newton Method
# The Truncated Newton method, also called *line search Newton conjugate gradient method*, is a method based on inexact Newton method and on line seasch. 
# At every step $k$, in order to compute the next direction $p^{(k)}$ we have to solve the linear system 
# 
# $\nabla ^2 f(x^{(k)})p = - \nabla f(x^{(k)})$
# 
# The idea behind truncated nm is to avoid solving this linear system exactly at every iteration by applying the CG method to it. 
# 
# The iterative solve is truncated either when the current solution is good enough (good approximation), or when we find the hessian to be non positive definite. 
# 
# In the first case the CG method has provided a descent direction. If the second case occurs, we start from the direction that caused the iterations to stop to compute a descent direction with line search (armijo + backtrack).
# 
# We guarantee that the direction $p$ chosen at each step $k$, is always a descent direction, even if $\nabla ^2 f(x^{(k)})$ is not posititive definite. 
# 
# Suggested termination condition for inner conjugate gradient method: $||r_k|| \leq \eta ||\nabla f(x_k)||$, where $r_k = \nabla ^2 f(x_k)p_k + \nabla f(x_k)$
# 
# ## Pseudocode
# - given $x^{(0)}$
# - for k = 1, 2, ...
#   - % solve linear system $B_k z = c_k$ (outer loop on NM)
#   - $z^{(0)} \leftarrow 0$
#   - $d^{(0)} \leftarrow - \nabla f(x^{(k)})$
#   - $r^{(0)} \leftarrow c_k$
#   - for j = 1, 2, ...
#     - % inner loop for solving $B_kz = c_k$
#     - if $d^{(j)T}B_kd^{(j)} > 0 $
#       - % proceed with conjugate gradient method
#       -  $\alpha ^{(j)} \leftarrow \frac{r^{(j)T}r^{(j)}}{d^{(j)T}B_kd^{(j)}}$
#       -  $z^{(j+1)} \leftarrow z^{(j)} + \alpha^{(j)}d^{(j)}$
#       -  $r^{(j+1)} \leftarrow r^{(j)} - \alpha^{(j)}B_kd^{(j)}$
#       -  $\beta ^{(j+1)} \leftarrow \frac{r^{(j+1)T}r^{(j+1)}}{r^{(j)T}r^{(j)}}$
#       -  $d^{(j+1)} \leftarrow r^{(j+1)} + \beta^{(j+1)}d^{(j)}$
#     - else
#       - % case in which $d^{(j)T}B_kd^{(j)} \leq 0 $
#       - if j = 0
#         - STOP returning $z = - \nabla f(x^{(k)})$
#       - else
#         - STOP returning $z = z^{(j)}$
#     - if $||B_kz-c_k|| \leq \eta_k ||c_k|| $
#       - STOP
#   - $p_{TN}^{(k)} \leftarrow z$
#   - $x^{(k+1)} \leftarrow x^{(k)} + \alpha ^{(k)}p_{TN}^{(k)}$ where $\alpha_k$ is computed with lineasearch (Armijo + backtracking
# 
# 
# #### Termination condition for the inner CG method:
# - $||r_k|| \leq min(0.5, \sqrt{\nabla f_k}) ||\nabla f(x_k)||$

# %% [markdown]
# ## Line search
# First we define the function that implements line search using armijo and backtracking.
# 
# - Choose an initial step length $\alpha_k^{(0)}$
# - For $j \geq 0$
#   - If $f(x_k + \alpha_k^{(j)} p_k) \leq f(x_k) + c_1 \alpha_k^{(j)} \nabla f(x_k)^Tp_k$
#     - accept $\alpha_k^{(j)}$
#   - Else
#     - $\alpha_k^{(j+1)} = \rho \alpha_k^{(j)}$
# 
#     
# $c_1 \in (0, 1)$ typically $c_1 = 10^{-4}$
# $\rho < 1$
# 
# 
# An acceptable step is always found as eventually $\alpha_k^{(j)}$ becomes small enough, but starting from a large values for $\alpha_k^{(0)}$ guarantees large step lengths whenever possible. For Newton methods it's crucial to choose $\alpha_k^{(0)} = 1$, to try to have quadratic convergence.

from scipy.linalg import cholesky_banded, cho_solve_banded, LinAlgError

# %%
def line_search(f, gradf, xk, p, alpha=1, rho=0.5, c1=1e-4):
    fxk = f(xk) #reduces function evaluations in loop
    grad_fxk = gradf(xk)
    while (f(xk + alpha * p) > fxk + (c1 * alpha * (grad_fxk @ p))):
        alpha *= rho
    return alpha

# %%
def forcing_term(gradient_norm, order_conv):
    if (order_conv == 'l'):
        return 0.5
    elif (order_conv == 'sl'):
        return min(0.5, np.sqrt(gradient_norm))
    else:
        return min(0.5, gradient_norm)
        

# %% [markdown]
# ### Preconditioning
# In case of ill conditioned matrices the inner loop of the conjugate gradient method  may need a massive number of iterations to converge. To solve this issue we use preconditioning. To preserve the sparsity of the matrix we take advantage of the pentadiagonal structure of the hessian and use the banded version of cholesky decomposition implemented by the function cholesky_banded. The choice of Cholesky over other decomposition techniques is essential to keep the symmetric structure of the matrix.
# 
# 

# %%
from scipy.linalg import solve_banded

# B: matrix 
# d: vector
# L: preconditioner (to be applied both on right and left)

def mat_vec_precond(B, d, L): 
    # full product to be computed: B_preconditioned * d 
    # equivalent to L^{-T} @ B @ L^{-1} @ d
    # divided into three sub operations, using solve_banded (keeping sparsity)

    # [L^{-1} @ d] found by solving linear system [L @ prod1 = d]
    prod1 = solve_banded((2, 0), L, d, overwrite_b=False)
    
    # [B @ prod1] computed directly
    prod2 = B @ prod1

    # [L^{-T} @ prod2] found by solving linear system [L^{T} @ prod2 = prod3]
    prod3 = solve_banded((0, 2), L, prod2, overwrite_b=True)
    return prod3

# %%
import numpy as np
from scipy.linalg import cholesky_banded

# solves the inner linear system Bz = c to find the descent direction pk_{tn} 
    # B = hessian(f)
    # c = - grad(f)
    # z = p

def inner_CG(B, c, z0, jmax, tol):

# Preconditioning Bz = c   ==>   B'y = c'
    B_banded = convert_to_banded(B)
    try: 
        L = cholesky_banded(B_banded, lower=True)
    except LinAlgError:    
        return c.copy()
    L = cholesky_banded(B_banded, lower=True)
    
    c_prime = solve_banded((0, 2), L, c, overwrite_b=False)
    # y = Rz (zeros)
    y = np.zeros_line(z0)
    
    # c' = L^{-t} * c by solving L^T c = c'

# initializations
    rk = c_prime.copy()
    dk = rk.copy()
    norm_c = np.linalg.norm(c_prime)
    relres = np.linalg.norm(rk) / norm_c
    j = 0

    
    while (j < jmax and relres > tol): 
        
        # B @ dk   ==>   B_prime @ dk 
        Bd_cond = mat_vec_precond(B, dk, L)

        # dk.T @ B @ dk    ==>   dk @ B_prime @ dk (not explicitely)   
        dBd_cond = dk @ Bd_cond
        
        # check if postitive definite
        # theoretically dBd > 0, but bc of machine precision we use a tolerance
        if (dBd_cond > 1e-12):      
                    
            alpha = (rk.T @ rk) / dBd_cond
            
            y = y + alpha * dk
            
            r_next = rk - alpha * Bd_cond
    
            beta = (r_next.T @ r_next)/(rk.T @ rk)
            dk = r_next + beta * dk
            rk = r_next
            
            relres = np.linalg.norm(rk) / norm_c
            j += 1

        else: # not positive definite
            if j == 0: # if it stopped at the first iteration
                return solve_banded((2,0), L, dk, overWrite_b=True)
            else: # if it stopped ater the first iteration
                return solve_banded((2,0), L, y, overWrite_b=True)
    return solve_banded((2,0), L, y, overWrite_b=True)

# %%
def truncated_newton(f, gradf, hessf, x0, tolgrad, kmax, jmax, order_conv='sl'):
    
    xk = x0.copy()
    path = [xk.copy()]
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
        path.append(xk.copy())

        grad_xk = gradf(xk)
        grad_xk_norm = np.linalg.norm(grad_xk)

    converges = grad_xk_norm < tolgrad
    return xk, grad_xk_norm, converges, k, path


# %%
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


