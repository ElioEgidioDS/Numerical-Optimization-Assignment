import numpy as np
from scipy.sparse import eye
from scipy.linalg import cholesky_banded, cho_solve_banded, LinAlgError
from Methods.Backtracking import Backtracking


def modified_newton(f, gradf, hessf, x0, tolgrad, kmax, order_conv, beta=1e-3):
    bcktrk = Backtracking(0.5,0.5,100)
    xk = x0.copy()
    n = np.size(xk)
    grad_xk = gradf(xk)
    grad_xk_norm = np.linalg.norm(grad_xk)
    k = 0

    while (k < kmax and grad_xk_norm >= tolgrad):
        k += 1
        Hess = hessf(xk)

        # check if function is positive definite with cholewsky decomposition
        # tau initialized t0 0 because if hessian is already positive definite 
        # we keep it as it is ,i.e. B_{k} = hessian
        tau = 0.0
        while True:
            I = eye(n, format='csc')
            E_k = tau * I
            B_k = Hess + E_k
            
            try: # factorize B_k using cholesky customized for banded matrices
                B_k_banded = convert_to_banded(B_k)
                R = cholesky_banded(B_k_banded)
                break
                
            # error catched if decomposition fails
            # i.e. if B_k is not positive definite
            except LinAlgError:
                tau = max(2*tau, beta)

        # solve linear system using incomplete cholesky factorization 
        # (already computed when testing positive definetness of B_k)
        p_mn = cho_solve_banded(R, -grad_xk)

        # find alpha using line search
        alpha = bcktrk.backtrack(p_mn,xk,f,1,gradf)
        # update current iterate
        xk = xk + alpha * p_mn
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