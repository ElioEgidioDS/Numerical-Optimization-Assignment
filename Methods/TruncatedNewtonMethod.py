import numpy as np
from scipy.sparse import issparse, csr_matrix, eye
from Methods.Backtracking import Backtracking


class TruncatedNewtonMethod:

    def __init__(self, tol, kmax, jmax, order_conv='sl', rho=0.5, c1=1e-4):
        self.tol = tol
        self.kmax = kmax
        self.jmax = jmax
        self.order_conv = order_conv
        self.rho = rho
        self.c1 = c1

        # Internal safeguards (do not change external interface)
        self._cg_curv_tol = 1e-14
        self._min_diag = 1e-12
        self._shift_base = 1e-3
        self._max_shift_tries = 5

    def line_search(self, f, grad_fxk, xk, p, alpha=1, rho=0.5, c1=1e-4):
        fxk = f(xk)
        slope = np.dot(grad_fxk, p)

        if slope >= 0:
            return 0.0

        while alpha > 1e-12:
            try:
                x_next = xk + alpha * p
                f_next = f(x_next)

                if np.isfinite(f_next) and (f_next <= fxk + (c1 * alpha * slope)):
                    return alpha

            except (OverflowError, ValueError, RuntimeWarning):
                pass

            alpha *= rho

        return 0.0

    def forcing_term(self, gradient_norm):
        # Forcing terms as in lab notes (Section 7.1)
        if self.order_conv == 'l':
            return 0.5
        elif self.order_conv == 'sl':
            return min(0.5, np.sqrt(gradient_norm))
        else:
            return min(0.5, gradient_norm)

    def _safe_csr(self, B):
        if issparse(B):
            return B.tocsr()
        return csr_matrix(B)

    def _jacobi_m_inv(self, B):
        # SPD preconditioner: M = diag(|B|)  (always positive)
        d = np.asarray(B.diagonal()).ravel()
        d = np.maximum(np.abs(d), self._min_diag)
        inv_d = 1.0 / d

        def apply(v):
            return inv_d * v

        return apply

    def _initial_shift(self, B):
        # Light diagonal shift
        dmin = float(np.min(B.diagonal()))
        if dmin > 0:
            return 0.0
        return self._shift_base - dmin
    
    def _gershgorin_shift(self, B):
        # For symmetric matrices: make Gershgorin lower bound positive.
        # tau >= -min_i (a_ii - sum_{j!=i} |a_ij|) + eps
        abs_row_sum = np.abs(B).sum(axis=1).A.ravel()
        diag = np.asarray(B.diagonal()).ravel()
        off = abs_row_sum - np.abs(diag)
        lower = diag - off
        min_lower = float(np.min(lower))
        if min_lower > 0:
            return 0.0
        return (-min_lower) + self._shift_base


    # PCG for (H + tau I) p = -g with relative residual stopping:
    # stop when ||r||/||c|| <= eta_k   (lab note about pcg stopping criteria)
    # Returns (p, cg_flag): "-", "NC", "MAX", "NAN"
    def inner_CG(self, B, c, eta_k):
        B = self._safe_csr(B)

        norm_c = np.linalg.norm(c)
        if norm_c < 1e-16:
            return np.zeros_like(c), "-"

        Minv = self._jacobi_m_inv(B)

        p = np.zeros_like(c)
        r = c.copy()  # since p0 = 0
        if not np.all(np.isfinite(r)):
            return np.zeros_like(c), "NAN"

        z = Minv(r)
        d = z.copy()
        rz = float(np.dot(r, z))

        for _ in range(self.jmax):
            Bd = B @ d
            dBd = float(np.dot(d, Bd))

            # negative curvature or loss of SPD
            if dBd <= self._cg_curv_tol:
                # return best available direction
                return p if np.linalg.norm(p) > 0 else c.copy(), "NC"

            alpha = rz / dBd
            p = p + alpha * d
            r = r - alpha * Bd

            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
                return p, "NAN"

            relres = np.linalg.norm(r) / norm_c
            if relres <= eta_k:
                return p, "-"

            z = Minv(r)
            rz_new = float(np.dot(r, z))
            if rz_new <= 0:
                return p, "NAN"

            beta = rz_new / rz
            d = z + beta * d
            rz = rz_new

        return p, "MAX"

    # NOTE: return_flag=False keeps the old 5-output signature.
    def truncated_newton(self, f, gradf, hessf, x0, return_flag=False):

        xk = x0.copy()
        xk_sequence = [xk.copy()]

        grad_xk = gradf(xk)
        grad_xk_norm = np.linalg.norm(grad_xk)

        flag_convergence = False
        flag = "MAX"  # default if we exit by max iterations
        bcktrk = Backtracking(self.c1, self.rho, 100)

        for k in range(self.kmax):

            if grad_xk_norm < self.tol:
                flag_convergence = True
                flag = "-"
                if return_flag:
                    return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence), flag
                return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence)

            if not np.all(np.isfinite(grad_xk)):
                flag = "NAN"
                break

            eta_k = self.forcing_term(grad_xk_norm)

            H = self._safe_csr(hessf(xk))
            I = eye(H.shape[0], format="csr")
            c = -grad_xk

            tau = max(self._initial_shift(H), self._gershgorin_shift(H))
            p_tn = None
            p_last = None
            cg_last = "MAX"
            cg_flag = "MAX"

            for _ in range(self._max_shift_tries):
                Hk = H + tau * I if tau > 0 else H
                p_try, cg_flag = self.inner_CG(Hk, c, eta_k)

                p_last = p_try
                cg_last = cg_flag

                if cg_flag in ("-", "MAX"):
                    p_tn = p_try
                    break
                
                if cg_flag == "NAN":
                    p_tn = p_try
                    flag = "NAN"
                    break
                
                # cg_flag == "NC": strengthen shift aggressively
                tau = max(10.0 * tau, self._shift_base) if tau > 0 else self._shift_base

            if p_tn is None:
                # DO NOT ABORT: use a safe fallback direction and keep going
                p_tn = p_last if p_last is not None else (-grad_xk)
                flag = "NC"


            # Ensure descent direction
            slope = float(np.dot(grad_xk, p_tn))
            if (not np.isfinite(slope)) or slope >= 0:
                p_tn = -grad_xk
                cg_flag = "NC"

            alpha = bcktrk.backtrack(p_tn, xk, f, 1.0, grad_xk)
            if alpha is None or alpha <= 0.0:
                # fallback: steepest descent
                p_sd = -grad_xk
                alpha_sd = bcktrk.backtrack(p_sd, xk, f, 1.0, grad_xk)

                if alpha_sd is None or alpha_sd <= 0.0:
                    flag = "LS"
                    break

                p_tn = p_sd
                alpha = alpha_sd

            x_next = xk + alpha * p_tn
            if not np.all(np.isfinite(x_next)):
                flag = "NAN"
                break

            xk = x_next
            xk_sequence.append(xk.copy())

            grad_xk = gradf(xk)
            grad_xk_norm = np.linalg.norm(grad_xk)

            if cg_flag == "NC" and flag == "MAX":
                flag = "NC"

        if return_flag:
            return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence), flag
        return xk, grad_xk_norm, flag_convergence, k, np.array(xk_sequence)
