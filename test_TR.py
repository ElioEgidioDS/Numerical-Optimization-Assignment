import os
import time
import numpy as np
import pandas as pd

from Problems.Problem_31 import Problem_31
from Problems.Problem_52 import Problem_52
from Problems.Problem_fd import Problem_fd
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from Methods.Finite_Differences import FiniteDifferences


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _compress_path(path_arr: np.ndarray) -> list:
    # for n == 2 keep points; for n > 2 store step lengths
    path_arr = np.asarray(path_arr)
    if path_arr.ndim == 2 and path_arr.shape[1] == 2:
        return path_arr.tolist()
    out = []
    for i in range(1, len(path_arr)):
        out.append(float(np.linalg.norm(path_arr[i] - path_arr[i - 1])))
    return out


def _aggregate_flag(flags_list):
    """
    Return a compact summary flag for a batch of runs:
    '-' if ALL are '-', otherwise the most frequent non-'-' flag (or 'MIX' if tie/empty).
    """
    if len(flags_list) == 0:
        return "MIX"
    if all(f == "-" for f in flags_list):
        return "-"
    non = [f for f in flags_list if f != "-"]
    if len(non) == 0:
        return "-"
    # most common non '-' (simple)
    vals, cnts = np.unique(non, return_counts=True)
    return str(vals[int(np.argmax(cnts))])


def _run_tr_exact(problem_class, x0_list, xRand_list, tr_solver, tag: str):
    # initial points
    init_rows = []
    for x0 in x0_list:
        p = problem_class(x0.shape[0])
        _log(f"[{tag}][exact] initial n={p.n} ...")

        t0 = time.time()
        x, ng, conv, it, path, flag = tr_solver.truncated_newton(
            p.function, p.gradient, p.hessian, x0, return_flag=True
        )
        dt = time.time() - t0

        _log(f"   -> conv={conv} it={it} ng={ng:.3e} flag={flag} time={dt:.2f}s")

        init_rows.append({
            "method": "Truncated Newton Initial Points",
            "setting": "exact",
            "tol": tr_solver.tol,
            "n": p.n,
            "time": dt,
            "iterations": int(it),
            "converges": bool(conv),
            "final_score": float(p.function(x)),
            "norm_gradient": float(ng),
            "flag": flag,
            "path": _compress_path(np.asarray(path)),
        })

    df_init = pd.DataFrame(init_rows)

    # random points (5 runs per size)
    rand_rows = []
    for block in xRand_list:
        n_dim = block.shape[1]
        p = problem_class(n_dim)
        _log(f"[{tag}][exact] random n={n_dim} ({len(block)} runs) ...")

        norm_grads, times, final_scores, iterations = [], [], [], []
        converges_list, flags, paths = [], [], []

        for i, x0 in enumerate(block, start=1):
            t0 = time.time()
            x, ng, conv, it, path, flag = tr_solver.truncated_newton(
                p.function, p.gradient, p.hessian, x0, return_flag=True
            )
            dt = time.time() - t0

            _log(f"   run {i}/{len(block)} -> conv={conv} it={it} ng={ng:.3e} flag={flag} time={dt:.2f}s")

            norm_grads.append(float(ng))
            times.append(dt)
            final_scores.append(float(p.function(x)))
            iterations.append(int(it))
            converges_list.append(bool(conv))
            flags.append(flag)
            paths.append(_compress_path(np.asarray(path)))

        rand_rows.append({
            "method": "Truncated Newton Random Points",
            "setting": "exact",
            "tol": tr_solver.tol,
            "n": n_dim,
            "time": float(np.mean(times)),
            "iterations": float(np.mean(iterations)),
            "converges": bool(np.all(converges_list)),
            "converges_count": int(np.sum(converges_list)),
            "final_score": float(np.mean(final_scores)),
            "norm_gradient": float(np.mean(norm_grads)),
            "flag": _aggregate_flag(flags),   # summary
            "flags": flags,                   # full list (len=5)
            "paths": paths,
        })

        _log(f"   -> converged {int(np.sum(converges_list))}/{len(converges_list)} | summary_flag={_aggregate_flag(flags)}")

    df_rand = pd.DataFrame(rand_rows)
    return df_init, df_rand


def _run_tr_fd(problem_class, x0_list, xRand_list, tr_solver, k_values, tag: str, fd_grad: bool):
    setting = "fd_grad_hess" if fd_grad else "fd_hess"

    # initial points (for each k)
    init_rows = []
    for x0 in x0_list:
        p = problem_class(x0.shape[0])
        fd = FiniteDifferences(p)

        for k in k_values:
            _log(f"[{tag}][{setting}] initial n={p.n} k={k} ...")

            if fd_grad:
                grad_fun = lambda x: fd.approximate_gradient(
                    x, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2
                )
            else:
                grad_fun = p.gradient

            hess_fun = lambda x: fd.approximate_hessian_pentadiag(
                x, grad_fun, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2
            )

            p_fd = Problem_fd(p, grad_fun, hess_fun)

            t0 = time.time()
            x, ng, conv, it, path, flag = tr_solver.truncated_newton(
                p_fd.function, p_fd.gradient, p_fd.hessian, x0, return_flag=True
            )
            dt = time.time() - t0

            _log(f"   -> conv={conv} it={it} ng={ng:.3e} flag={flag} time={dt:.2f}s")

            init_rows.append({
                "method": "Truncated Newton Initial Points",
                "setting": setting,
                "tol": tr_solver.tol,
                "n": p.n,
                "k": k,
                "time": dt,
                "iterations": int(it),
                "converges": bool(conv),
                "final_score": float(p.function(x)),   # score on true problem
                "norm_gradient": float(ng),
                "flag": flag,
                "path": _compress_path(np.asarray(path)),
            })

    df_init = pd.DataFrame(init_rows)

    # random points (for each n, for each k)
    rand_rows = []
    for block in xRand_list:
        n_dim = block.shape[1]
        p = problem_class(n_dim)
        fd = FiniteDifferences(p)

        for k in k_values:
            _log(f"[{tag}][{setting}] random n={n_dim} k={k} ({len(block)} runs) ...")

            if fd_grad:
                grad_fun = lambda x: fd.approximate_gradient(
                    x, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2
                )
            else:
                grad_fun = p.gradient

            hess_fun = lambda x: fd.approximate_hessian_pentadiag(
                x, grad_fun, k, step_mode="adaptive", x_ref=x, zero_floor=1e-2
            )

            p_fd = Problem_fd(p, grad_fun, hess_fun)

            norm_grads, times, final_scores, iterations = [], [], [], []
            converges_list, flags, paths = [], [], []

            for i, x0 in enumerate(block, start=1):
                t0 = time.time()
                x, ng, conv, it, path, flag = tr_solver.truncated_newton(
                    p_fd.function, p_fd.gradient, p_fd.hessian, x0, return_flag=True
                )
                dt = time.time() - t0

                _log(f"   run {i}/{len(block)} -> conv={conv} it={it} ng={ng:.3e} flag={flag} time={dt:.2f}s")

                norm_grads.append(float(ng))
                times.append(dt)
                final_scores.append(float(p.function(x)))  # score on true problem
                iterations.append(int(it))
                converges_list.append(bool(conv))
                flags.append(flag)
                paths.append(_compress_path(np.asarray(path)))

            rand_rows.append({
                "method": "Truncated Newton Random Points",
                "setting": setting,
                "tol": tr_solver.tol,
                "n": n_dim,
                "k": k,
                "time": float(np.mean(times)),
                "iterations": float(np.mean(iterations)),
                "converges": bool(np.all(converges_list)),
                "converges_count": int(np.sum(converges_list)),
                "final_score": float(np.mean(final_scores)),
                "norm_gradient": float(np.mean(norm_grads)),
                "flag": _aggregate_flag(flags),  # summary
                "flags": flags,                  # full list
                "paths": paths,
            })

            _log(f"   -> converged {int(np.sum(converges_list))}/{len(converges_list)} | summary_flag={_aggregate_flag(flags)}")

    df_rand = pd.DataFrame(rand_rows)
    return df_init, df_rand


def run_problem(problem_class, out_root: str, tag: str):
    _log(f"\n==============================")
    _log(f"START {tag} -> saving in {out_root}")
    _log(f"==============================\n")

    _ensure_dir(out_root)

    # Subfolders
    out_exact = os.path.join(out_root, "exact")
    out_fdh   = os.path.join(out_root, "fd_hess")
    out_fdgh  = os.path.join(out_root, "fd_grad_hess")
    _ensure_dir(out_exact)
    _ensure_dir(out_fdh)
    _ensure_dir(out_fdgh)

    # Settings (same spirit as your scripts)
    np.random.seed(352283)

    tol = 1e-6
    kmax = 1000
    jmax = 1000
    order_conv = "sl"
    rho = 0.6
    c1 = 1e-4

    n_list = [2, 10**3, 10**4, 10**5]
    runs_per_n = 5

    # initial points = recommended x0 from the problem
    x0_list = [problem_class(n).x0.copy() for n in n_list]

    # random blocks: Uniform in [x0 - 1, x0 + 1] component-wise
    xRand_list = []
    for n in n_list:
        xbar = problem_class(n).x0.copy()
        low = xbar - 1.0
        high = xbar + 1.0
        xRand_list.append(np.random.uniform(low=low, high=high, size=(runs_per_n, n)))

    # solver
    tr = TruncatedNewtonMethod(tol, kmax, jmax, order_conv, rho, c1)

    # 1) exact
    df_init, df_rand = _run_tr_exact(problem_class, x0_list, xRand_list, tr, tag=tag)
    df_init.to_csv(os.path.join(out_exact, "x_initial_tr.csv"), index=False)
    df_rand.to_csv(os.path.join(out_exact, "x_random_tr.csv"), index=False)
    _log(f"[{tag}] saved: {out_exact}")

    # 2) fd_hess
    k_values = [4, 8, 12]
    df_init_fdh, df_rand_fdh = _run_tr_fd(problem_class, x0_list, xRand_list, tr, k_values, tag=tag, fd_grad=False)
    df_init_fdh.to_csv(os.path.join(out_fdh, "x_initial_tr.csv"), index=False)
    df_rand_fdh.to_csv(os.path.join(out_fdh, "x_random_tr.csv"), index=False)
    _log(f"[{tag}] saved: {out_fdh}")

    # 3) fd_grad_hess
    df_init_fdgh, df_rand_fdgh = _run_tr_fd(problem_class, x0_list, xRand_list, tr, k_values, tag=tag, fd_grad=True)
    df_init_fdgh.to_csv(os.path.join(out_fdgh, "x_initial_tr.csv"), index=False)
    df_rand_fdgh.to_csv(os.path.join(out_fdgh, "x_random_tr.csv"), index=False)
    _log(f"[{tag}] saved: {out_fdgh}")

    _log(f"\nDONE {tag}\n")


if __name__ == "__main__":
    run_problem(Problem_31, os.path.join("csv", "test_tr", "p31_test"), tag="p31")
    run_problem(Problem_52, os.path.join("csv", "test_tr", "p52_test"), tag="p52")
