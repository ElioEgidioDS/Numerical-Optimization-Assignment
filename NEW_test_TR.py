# test_TR.py
# Runs Truncated Newton only (NEW_TNM.py version) on Problem 31 and 52
# for 3 settings: exact, fd_hess, fd_grad_hess
# Prints progress to console (no CSV saving)

import time
import numpy as np

from Problems.Problem_31 import Problem_31
from Problems.Problem_52 import Problem_52
from Problems.Problem_fd import Problem_fd
from Methods.Finite_Differences import FiniteDifferences

# IMPORTANT: import the NEW truncated newton (the one with 5 outputs, no return_flag)
from Methods.NEW_TNM import TruncatedNewtonMethod


def _log(msg: str) -> None:
    print(msg, flush=True)


def _compress_path(path_arr: np.ndarray) -> list:
    """
    For n == 2 keep points; for n > 2 store step lengths.
    """
    path_arr = np.asarray(path_arr)
    if path_arr.ndim == 2 and path_arr.shape[1] == 2:
        return path_arr.tolist()
    out = []
    for i in range(1, len(path_arr)):
        out.append(float(np.linalg.norm(path_arr[i] - path_arr[i - 1])))
    return out


def _run_tr(tr_solver, f, gradf, hessf, x0):
    """
    Adapter for NEW_TNM.TruncatedNewtonMethod:
    returns: x, ng, conv, it, path
    """
    return tr_solver.truncated_newton(f, gradf, hessf, x0)


def _run_exact(problem_class, x0_list, xRand_list, tr_solver, tag: str):
    # initial points
    for x0 in x0_list:
        p = problem_class(x0.shape[0])
        _log(f"[{tag}][exact] initial n={p.n} ...")

        t0 = time.time()
        x, ng, conv, it, path = _run_tr(tr_solver, p.function, p.gradient, p.hessian, x0)
        dt = time.time() - t0

        _log(f"   -> conv={conv} it={it} ng={ng:.3e} time={dt:.2f}s")

    # random points (runs per size)
    for block in xRand_list:
        n_dim = block.shape[1]
        p = problem_class(n_dim)
        _log(f"[{tag}][exact] random n={n_dim} ({len(block)} runs) ...")

        conv_count = 0
        for i, x0 in enumerate(block, start=1):
            t0 = time.time()
            x, ng, conv, it, path = _run_tr(tr_solver, p.function, p.gradient, p.hessian, x0)
            dt = time.time() - t0

            conv_count += int(bool(conv))
            _log(f"   run {i}/{len(block)} -> conv={conv} it={it} ng={ng:.3e} time={dt:.2f}s")

        _log(f"   -> converged {conv_count}/{len(block)}")


def _run_fd(problem_class, x0_list, xRand_list, tr_solver, k_values, tag: str, fd_grad: bool):
    setting = "fd_grad_hess" if fd_grad else "fd_hess"

    # initial points
    for x0 in x0_list:
        p = problem_class(x0.shape[0])
        fd = FiniteDifferences(p)

        for k in k_values:
            _log(f"[{tag}][{setting}] initial n={p.n} k={k} ...")

            if fd_grad:
                grad_fun = lambda x, kk=k: fd.approximate_gradient(
                    x, kk, step_mode="adaptive", x_ref=x, zero_floor=1e-2
                )
            else:
                grad_fun = p.gradient

            hess_fun = lambda x, g=grad_fun, kk=k: fd.approximate_hessian_pentadiag(
                x, g, kk, step_mode="adaptive", x_ref=x, zero_floor=1e-2
            )

            p_fd = Problem_fd(p, grad_fun, hess_fun)

            t0 = time.time()
            x, ng, conv, it, path = _run_tr(tr_solver, p_fd.function, p_fd.gradient, p_fd.hessian, x0)
            dt = time.time() - t0

            _log(f"   -> conv={conv} it={it} ng={ng:.3e} time={dt:.2f}s")

    # random points
    for block in xRand_list:
        n_dim = block.shape[1]
        p = problem_class(n_dim)
        fd = FiniteDifferences(p)

        for k in k_values:
            _log(f"[{tag}][{setting}] random n={n_dim} k={k} ({len(block)} runs) ...")

            if fd_grad:
                grad_fun = lambda x, kk=k: fd.approximate_gradient(
                    x, kk, step_mode="adaptive", x_ref=x, zero_floor=1e-2
                )
            else:
                grad_fun = p.gradient

            hess_fun = lambda x, g=grad_fun, kk=k: fd.approximate_hessian_pentadiag(
                x, g, kk, step_mode="adaptive", x_ref=x, zero_floor=1e-2
            )

            p_fd = Problem_fd(p, grad_fun, hess_fun)

            conv_count = 0
            for i, x0 in enumerate(block, start=1):
                t0 = time.time()
                x, ng, conv, it, path = _run_tr(tr_solver, p_fd.function, p_fd.gradient, p_fd.hessian, x0)
                dt = time.time() - t0

                conv_count += int(bool(conv))
                _log(f"   run {i}/{len(block)} -> conv={conv} it={it} ng={ng:.3e} time={dt:.2f}s")

            _log(f"   -> converged {conv_count}/{len(block)}")


def run_problem(problem_class, tag: str):
    _log(f"\n==============================")
    _log(f"START {tag}")
    _log(f"==============================\n")

    # Settings
    np.random.seed(352283)

    tol = 1e-6
    kmax = 1000
    jmax = 1000
    order_conv = "sl"
    rho = 0.6
    c1 = 1e-4

    n_list = [2, 10**3, 10**4, 10**5]
    runs_per_n = 5
    rand_low, rand_high = -2.0, 2.0

    # initial points from each problem's recommended x0
    x0_list = [problem_class(n).x0.copy() for n in n_list]

    # random blocks: Uniform in [x0 - 1, x0 + 1] component-wise
    xRand_list = []
    for n in n_list:
        xbar = problem_class(n).x0.copy()
        low = xbar - 1.0
        high = xbar + 1.0
        xRand_list.append(np.random.uniform(low=low, high=high, size=(runs_per_n, n)))

    # solver (NEW_TNM)
    tr = TruncatedNewtonMethod(tol, kmax, jmax, order_conv, rho, c1)

    # 1) exact
    _run_exact(problem_class, x0_list, xRand_list, tr, tag=tag)

    # 2) fd_hess
    k_values = [4, 8, 12]
    _run_fd(problem_class, x0_list, xRand_list, tr, k_values, tag=tag, fd_grad=False)

    # 3) fd_grad_hess
    _run_fd(problem_class, x0_list, xRand_list, tr, k_values, tag=tag, fd_grad=True)

    _log(f"\nDONE {tag}\n")


if __name__ == "__main__":
    run_problem(Problem_31, tag="p31")
    run_problem(Problem_52, tag="p52")
