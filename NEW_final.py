import os
import time
import numpy as np
import pandas as pd

from Methods.ModifiedNewtonMethod import ModifiedNewtonMethod
from Methods.TruncatedNewtonMethod import TruncatedNewtonMethod
from Methods.Finite_Differences import FiniteDifferences
from Problems.Problem_fd import Problem_fd
import sys

_T0 = time.time()

def log(msg: str):
    dt = time.time() - _T0
    print(f"[{dt:9.2f}s] {msg}", flush=True)


def calculate_convergence_order(step_sizes):
    """
    experimental order of convergence q from a list of step sizes:
        s_k = ||x_{k+1} - x_k||
        q_k = log(s_{k+1}/s_k) / log(s_k/s_{k-1})

    returns the last available q, or nan if not computable.
    """
    if step_sizes is None:
        return np.nan

    s = np.asarray(step_sizes, dtype=float)
    if s.size < 3:
        return np.nan
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        return np.nan

    ratios = s[1:] / s[:-1]
    if ratios.size < 2:
        return np.nan
    if np.any(ratios <= 0):
        return np.nan

    q_vals = []
    for k in range(1, ratios.size):
        num = np.log(ratios[k])
        den = np.log(ratios[k - 1])
        if np.isfinite(num) and np.isfinite(den) and den != 0:
            q_vals.append(num / den)

    return q_vals[-1] if q_vals else np.nan


def _ensure_dir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def _get_problem_name(problem_instance):
    if hasattr(problem_instance, "name"):
        return str(problem_instance.name)
    return type(problem_instance).__name__


def _unpack_nm_output(out):
    """
    supports both old (5) and new (6) return formats:
      new: x, path, grad_norm, success, iters, flag
      old: x, path, grad_norm, success, iters
    """
    if isinstance(out, tuple) or isinstance(out, list):
        if len(out) == 6:
            return out[0], out[1], out[2], out[3], out[4], out[5]
        if len(out) == 5:
            return out[0], out[1], out[2], out[3], out[4], "x"
    raise ValueError("unexpected return format from ModifiedNewtonMethod.modified_newton")


def _unpack_tr_output(out):
    """
    supports both old (5) and new (6) return formats:
      new: x, grad_norm, success, iters, path, flag
      old: x, grad_norm, success, iters, path
    """
    if isinstance(out, tuple) or isinstance(out, list):
        if len(out) == 6:
            return out[0], out[1], out[2], out[3], out[4], out[5]
        if len(out) == 5:
            return out[0], out[1], out[2], out[3], out[4], "x"
    raise ValueError("unexpected return format from TruncatedNewtonMethod.truncated_newton")


def _steps_from_path(path):
    """
    path is expected as array (m,n) of iterates.
    returns list of step norms ||x_{k+1}-x_k||.
    """
    if path is None:
        return None
    P = np.asarray(path)
    if P.ndim != 2 or P.shape[0] < 2:
        return None
    diffs = P[1:] - P[:-1]
    return np.linalg.norm(diffs, axis=1).tolist()


def _append_paths_n2(path_rows, run_id, problem_name, n, method, case, h_mode, k_fd, tol, start_type, point_id, path):
    """
    saves only n=2 full iterate matrix into the path table.
    required columns (plus some extra for filtering):
      Problem,Method,Case,h_mode,path_id,k,x1,x2
    """
    if n != 2 or path is None:
        return

    P = np.asarray(path)
    if P.ndim != 2 or P.shape[1] != 2:
        return

    for k_iter in range(P.shape[0]):
        path_rows.append({
            "Problem": problem_name,
            "Method": method,
            "Case": case,
            "h_mode": h_mode,
            "k_fd": k_fd,
            "tol": tol,
            "start_type": start_type,
            "path_id": point_id,
            "run_id": run_id,
            "k": int(k_iter),
            "x1": float(P[k_iter, 0]),
            "x2": float(P[k_iter, 1]),
        })

def _append_step_norms(norm_rows, run_id, problem_name, n, method, case, h_mode, k_fd, tol, start_type, point_id, path):
    """
    Saves the step norms ||x_{k+1} - x_k|| into a list of dicts.
    """
    if path is None:
        return

    # Helper from your existing code to get step sizes
    steps = _steps_from_path(path)
    
    if steps is None or len(steps) == 0:
        return

    for k_step, step_val in enumerate(steps):
        norm_rows.append({
            "Problem": problem_name,
            "Size":n,
            "Method": method,
            "Case": case,
            "h_mode": h_mode,
            "k_fd": k_fd,
            "tol": tol,
            "start_type": start_type,
            "path_id": point_id,
            "run_id": run_id,
            "k": int(k_step),            
            "step_norm": float(step_val) 
        })

def _run_one_nm(nm, problem_proxy, problem_base, x0):
    t0 = time.time()
    out = nm.modified_newton(problem_proxy, x0)
    tsec = time.time() - t0

    x, path, grad_norm, success, iters, flag = _unpack_nm_output(out)
    fval = problem_base.function(x)
    q = calculate_convergence_order(_steps_from_path(path))

    path_len = int(np.asarray(path).shape[0]) if path is not None else 0
    last_step = np.nan
    steps = _steps_from_path(path)
    if steps is not None and len(steps) > 0:
        last_step = float(steps[-1])

    return x, path, grad_norm, bool(success), int(iters), flag, float(tsec), float(fval), q, path_len, last_step


def _run_one_tr(tr, problem_proxy, problem_base, x0):
    t0 = time.time()
    out = tr.truncated_newton(problem_proxy.function, problem_proxy.gradient, problem_proxy.hessian, x0)
    tsec = time.time() - t0

    x, grad_norm, success, iters, path, flag = _unpack_tr_output(out)
    fval = problem_base.function(x)
    q = calculate_convergence_order(_steps_from_path(path))

    path_len = int(np.asarray(path).shape[0]) if path is not None else 0
    last_step = np.nan
    steps = _steps_from_path(path)
    if steps is not None and len(steps) > 0:
        last_step = float(steps[-1])

    return x, path, grad_norm, bool(success), int(iters), flag, float(tsec), float(fval), q, path_len, last_step


def _save_csv(df, filepath, append=False):
    _ensure_dir(os.path.dirname(filepath))
    if append and os.path.exists(filepath):
        df.to_csv(filepath, index=False, mode="a", header=False)
    else:
        df.to_csv(filepath, index=False)


def final(
    x0,
    xRand,
    problem_main,
    tol=1e-6,
    max_iter_nm=1000,
    max_iter_tr=1000,
    inner_max_iter_tr=1000,
    rho=0.6,
    c1=1e-4,
    k_values=(4, 8, 12),
    h_modes=("scalar", "adaptive"),
    zero_floor=1e-2,
    cases=("Exact", "Mixed FD", "Full FD"),
    out_final_csv=os.path.join("csv", "final", "final_results.csv"),
    out_paths_csv=os.path.join("csv", "path", "paths_n2.csv"),
    out_norms_csv=os.path.join("csv", "norms", "step_norms.csv"),
    append_to_existing=False,
):
    """
    creates:
      1) csv/final/final_results.csv
         one row = one run (no averaging of the 5 random points)
      2) csv/path/paths_n2.csv
         only n=2 iterates for every run (initial + random), all cases/methods

    columns in final_results.csv (you can add/remove safely later):
      Problem,n,Method,Case,h_mode,k_fd,tol,start_type,point_id,run_id,
      GradNorm,Iterations,MaxIterations,InnerMaxIterations,Success,flag,
      ConvergenceRate,TimeSeconds,FinalF,PathLen,LastStep
    """

    # ----------------------------
    # light logging (prints with elapsed seconds)
    # ----------------------------
    _t0 = time.time()

    def log(msg: str):
        dt = time.time() - _t0
        print(f"[{dt:9.2f}s] {msg}", flush=True)

    problem_name = _get_problem_name(problem_main)

    # detect dimensions from x0 and xRand blocks
    dims = set([int(p.shape[0]) for p in x0]) if x0 is not None else set()
    if xRand is not None:
        dims |= set([int(X.shape[1]) for X in xRand])
    dims = sorted(dims)

    log(f"START final() | Problem={problem_name} | dims={dims} | cases={cases} | tol={tol}")
    log(f"Output: final='{out_final_csv}' | paths='{out_paths_csv}' | append={append_to_existing}")

    nm = ModifiedNewtonMethod(tol, max_iter_nm, rho, c1)
    tr = TruncatedNewtonMethod(tol, max_iter_tr, inner_max_iter_tr, "sl", rho, c1)

    rows = []
    path_rows = []
    norm_rows = []
    run_id_counter = 0

    def get_xbar(n):
        for p in x0:
            if int(p.shape[0]) == int(n):
                return p
        return None

    def get_rand_block(n):
        for X in xRand:
            if int(X.shape[1]) == int(n):
                return X
        return None

    def add_result_row(method, case, h_mode, k_fd, start_type, point_id, run_id,
                       grad_norm, iters, success, flag, q, tsec, fval, path_len, last_step):
        rows.append({
            "Problem": problem_name,
            "n": int(n),
            "Method": method,
            "Case": case,
            "h_mode": h_mode,
            "k_fd": k_fd,
            "tol": tol,
            "start_type": start_type,
            "point_id": point_id,
            "run_id": int(run_id),
            "GradNorm": float(grad_norm),
            "Iterations": int(iters),
            "MaxIterations": int(max_iter_nm) if method == "nm" else int(max_iter_tr),
            "InnerMaxIterations": np.nan if method == "nm" else int(inner_max_iter_tr),
            "Success": bool(success),
            "flag": flag,
            "ConvergenceRate": q,
            "TimeSeconds": float(tsec),
            "FinalF": float(fval),
            "PathLen": int(path_len),
            "LastStep": last_step,
        })

    for n in dims:
        log(f"--- n={n} | building base problem + FD ---")
        problem_base = type(problem_main)(n)
        fd = FiniteDifferences(problem_base)

        xbar = get_xbar(n)
        Xrand = get_rand_block(n)

        starts = []
        if xbar is not None:
            starts.append(("initial", "xbar", xbar))
        if Xrand is not None:
            for i in range(Xrand.shape[0]):
                starts.append(("random", f"rand{i+1}", Xrand[i, :]))

        if not starts:
            log(f"n={n} | no starting points found -> skipping")
            continue

        log(f"n={n} | starts={len(starts)} (" + ", ".join([f"{st}:{pid}" for st, pid, _ in starts]) + ")")

        # ----------------------------
        # Exact case
        # ----------------------------
        if "Exact" in cases:
            case = "Exact"
            log(f"[CASE] {case} | n={n} | runs={len(starts)} | methods=nm,tr")

            for start_type, point_id, xstart in starts:
                run_id_counter += 1
                run_id = run_id_counter

                # Newton
                x, path, gnorm, ok, iters, flag, tsec, fval, q, plen, last_step = _run_one_nm(
                    nm, problem_base, problem_base, xstart
                )
                add_result_row("nm", case, "", np.nan, start_type, point_id, run_id,
                               gnorm, iters, ok, flag, q, tsec, fval, plen, last_step)
                
                _append_paths_n2(path_rows, run_id, problem_name, n, "nm", case, "", np.nan, tol, start_type, point_id, path)

                if ok:
                    _append_step_norms(norm_rows, run_id, problem_name, n, "nm", case, "", np.nan, tol, start_type, point_id, path)

                log(f"Done run={run_id} | {case} | n={n} | start={start_type}:{point_id} | nm | ok={ok} | iters={iters} | ||g||={gnorm:.3e} | f={fval:.3e} | t={tsec:.2f}s | flag={flag}")

                # Truncated Newton
                x, path, gnorm, ok, iters, flag, tsec, fval, q, plen, last_step = _run_one_tr(
                    tr, problem_base, problem_base, xstart
                )
                add_result_row("tr", case, "", np.nan, start_type, point_id, run_id,
                               gnorm, iters, ok, flag, q, tsec, fval, plen, last_step)
                
                _append_paths_n2(path_rows, run_id, problem_name, n, "tr", case, "", np.nan, tol, start_type, point_id, path)
                if ok:
                    _append_step_norms(norm_rows, run_id, problem_name, n,"tr", case, "", np.nan, tol, start_type, point_id, path)

                log(f"Done run={run_id} | {case} | n={n} | start={start_type}:{point_id} | tr | ok={ok} | iters={iters} | ||g||={gnorm:.3e} | f={fval:.3e} | t={tsec:.2f}s | flag={flag}")

        # ----------------------------
        # Mixed FD case (exact grad + fd hessian)
        # ----------------------------
        if "Mixed FD" in cases:
            case = "Mixed FD"
            log(f"[CASE] {case} | n={n} | combos={len(h_modes)*len(k_values)} | runs_per_combo={len(starts)} | methods=nm,tr")

            for h_mode in h_modes:
                for k_fd in k_values:
                    log(f"[CFG] {case} | n={n} | h_mode={h_mode} | k_fd={k_fd}")

                    grad_fun = problem_base.gradient

                    def hess_fun(x, gf=grad_fun, kk=k_fd, mm=h_mode):
                        return fd.approximate_hessian_pentadiag(
                            x,
                            grad_fun=gf,
                            k_step=kk,
                            step_mode=mm,
                            x_ref=None,
                            zero_floor=zero_floor
                        )

                    problem_mixed = Problem_fd(problem_base, grad_fun, hess_fun)

                    for start_type, point_id, xstart in starts:
                        run_id_counter += 1
                        run_id = run_id_counter

                        # Newton
                        x, path, gnorm, ok, iters, flag, tsec, fval, q, plen, last_step = _run_one_nm(
                            nm, problem_mixed, problem_base, xstart
                        )
                        add_result_row("nm", case, h_mode, int(k_fd), start_type, point_id, run_id,
                                       gnorm, iters, ok, flag, q, tsec, fval, plen, last_step)
                        
                        _append_paths_n2(path_rows, run_id, problem_name, n, "nm", case, h_mode, int(k_fd), tol, start_type, point_id, path)
                        if ok:
                            _append_step_norms(norm_rows, run_id, problem_name,n, "nm", case, h_mode, int(k_fd), tol, start_type, point_id, path)

                        log(f"Done run={run_id} | {case} | n={n} | {h_mode},k={k_fd} | start={start_type}:{point_id} | nm | ok={ok} | iters={iters} | ||g||={gnorm:.3e} | f={fval:.3e} | t={tsec:.2f}s | flag={flag}")

                        # Truncated Newton
                        x, path, gnorm, ok, iters, flag, tsec, fval, q, plen, last_step = _run_one_tr(
                            tr, problem_mixed, problem_base, xstart
                        )
                        add_result_row("tr", case, h_mode, int(k_fd), start_type, point_id, run_id,
                                       gnorm, iters, ok, flag, q, tsec, fval, plen, last_step)
                        
                        _append_paths_n2(path_rows, run_id, problem_name, n, "tr", case, h_mode, int(k_fd), tol, start_type, point_id, path)
                        if ok:
                            _append_step_norms(norm_rows, run_id, problem_name, n,"tr", case, h_mode, int(k_fd), tol, start_type, point_id, path)

                        log(f"Done run={run_id} | {case} | n={n} | {h_mode},k={k_fd} | start={start_type}:{point_id} | tr | ok={ok} | iters={iters} | ||g||={gnorm:.3e} | f={fval:.3e} | t={tsec:.2f}s | flag={flag}")

        # ----------------------------
        # Full FD case (fd grad + fd hessian)
        # ----------------------------
        if "Full FD" in cases:
            case = "Full FD"
            log(f"[CASE] {case} | n={n} | combos={len(h_modes)*len(k_values)} | runs_per_combo={len(starts)} | methods=nm,tr")

            for h_mode in h_modes:
                for k_fd in k_values:
                    log(f"[CFG] {case} | n={n} | h_mode={h_mode} | k_fd={k_fd}")

                    grad_fun = fd.make_grad_fun(
                        k_step=int(k_fd),
                        step_mode=h_mode,
                        scheme="centered",
                        x_ref=None,
                        zero_floor=zero_floor
                    )

                    def hess_fun(x, gf=grad_fun, kk=k_fd, mm=h_mode):
                        return fd.approximate_hessian_pentadiag(
                            x,
                            grad_fun=gf,
                            k_step=int(kk),
                            step_mode=mm,
                            x_ref=None,
                            zero_floor=zero_floor
                        )

                    problem_full = Problem_fd(problem_base, grad_fun, hess_fun)

                    for start_type, point_id, xstart in starts:
                        run_id_counter += 1
                        run_id = run_id_counter

                        # Newton
                        x, path, gnorm, ok, iters, flag, tsec, fval, q, plen, last_step = _run_one_nm(
                            nm, problem_full, problem_base, xstart
                        )
                        add_result_row("nm", case, h_mode, int(k_fd), start_type, point_id, run_id,
                                       gnorm, iters, ok, flag, q, tsec, fval, plen, last_step)
                        
                        _append_paths_n2(path_rows, run_id, problem_name, n, "nm", case, h_mode, int(k_fd), tol, start_type, point_id, path)
                        if ok:
                            _append_step_norms(norm_rows, run_id, problem_name, n,"nm", case, h_mode, int(k_fd), tol, start_type, point_id, path)

                        log(f"Done run={run_id} | {case} | n={n} | {h_mode},k={k_fd} | start={start_type}:{point_id} | nm | ok={ok} | iters={iters} | ||g||={gnorm:.3e} | f={fval:.3e} | t={tsec:.2f}s | flag={flag}")

                        # Truncated Newton
                        x, path, gnorm, ok, iters, flag, tsec, fval, q, plen, last_step = _run_one_tr(
                            tr, problem_full, problem_base, xstart
                        )
                        add_result_row("tr", case, h_mode, int(k_fd), start_type, point_id, run_id,
                                       gnorm, iters, ok, flag, q, tsec, fval, plen, last_step)
                        
                        _append_paths_n2(path_rows, run_id, problem_name, n, "tr", case, h_mode, int(k_fd), tol, start_type, point_id, path)
                        if ok:
                            _append_step_norms(norm_rows, run_id, problem_name, n,"tr", case, h_mode, int(k_fd), tol, start_type, point_id, path)

                        log(f"Done run={run_id} | {case} | n={n} | {h_mode},k={k_fd} | start={start_type}:{point_id} | tr | ok={ok} | iters={iters} | ||g||={gnorm:.3e} | f={fval:.3e} | t={tsec:.2f}s | flag={flag}")

    df_final = pd.DataFrame(rows)
    df_paths = pd.DataFrame(path_rows)
    df_norms = pd.DataFrame(norm_rows)

    log(f"Saving CSVs... final_rows={len(df_final)} | paths_rows={len(df_paths)}")
    _save_csv(df_final, out_final_csv, append=append_to_existing)
    _save_csv(df_paths, out_paths_csv, append=append_to_existing)
    _save_csv(df_norms, out_norms_csv, append=append_to_existing)

    log("END final()")
    return df_final, df_paths, df_norms


def build_starting_points(problem_class, n_list, runs_per_n=5, seed=352283):
    """
    Returns:
      x0_list: list of np.array, one per n, the recommended initial point (xbar)
      xRand_list: list of np.array, one per n, shape (runs_per_n, n)
                 random points sampled in [xbar-1, xbar+1] component-wise
    """
    np.random.seed(seed)

    x0_list = []
    xRand_list = []

    for n in n_list:
        prob = problem_class(n)
        xbar = prob.x0.copy()
        x0_list.append(xbar)

        low = xbar - 1.0
        high = xbar + 1.0
        Xrand = np.random.uniform(low=low, high=high, size=(runs_per_n, n))
        xRand_list.append(Xrand)

    return x0_list, xRand_list


def main():
    from Problems.Problem_52 import Problem_52
    from Problems.Problem_31 import Problem_31

    problems = [Problem_52, Problem_31]
    n_list = [2, 10**3, 10**4, 10**5]  # prova con n piccoli

    out_final = os.path.join("csv", "final", "final_results.csv")
    out_paths = os.path.join("csv", "path", "paths_n2.csv")
    out_norms = os.path.join("csv", "norms", "convergence_errors.csv")

    for idx, problem_class in enumerate(problems):
        x0, xRand = build_starting_points(problem_class, n_list, runs_per_n=5, seed=352283)
        problem_main = problem_class(n_list[0])

        df_final, df_paths , df_norms = final(
            x0=x0,
            xRand=xRand,
            problem_main=problem_main,
            tol=1e-6,
            max_iter_nm=1000,
            max_iter_tr=1000,
            inner_max_iter_tr=1000,
            rho=0.6,
            c1=1e-4,
            k_values=(4, 8, 12),
            h_modes=("scalar", "adaptive"),
            zero_floor=1e-2,
            cases=("Exact", "Mixed FD", "Full FD"),
            out_final_csv=out_final,
            out_paths_csv=out_paths,
            out_norms_csv=out_norms,
            append_to_existing=(idx > 0),  # prima volta sovrascrive, poi appende
        )

        print(f"Done {problem_class.__name__}: final rows {len(df_final)}, path rows {len(df_paths)}")

    print("Saved:")
    print(" -", out_final)
    print(" -", out_paths)
    print(" -", out_norms)


if __name__ == "__main__":
    main()
