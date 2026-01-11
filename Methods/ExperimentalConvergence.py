"""
Experimental (a-posteriori) convergence orders and plotting utilities.

Reference: laboratories file, section 7.2 (Orders of Convergence).
We estimate the (local) order q without knowing x* using the iterate differences:

    ê^(k) = x^(k) - x^(k-1)

and for k sufficiently large:

    q_k ≈ log( ||ê^(k+1)|| / ||ê^(k)|| ) / log( ||ê^(k)|| / ||ê^(k-1)|| )

This module provides:
  1) experimental_orders_from_xk: compute the sequence {q_k} from a path {x_k}.
  2) experimental_rate_summary: estimate a scalar q from the tail of q_k.
  3) plot_convergence_orders: readable matplotlib plotting helper.
  4) High-level figure generation: group runs and save figures with required naming.

"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


ArrayLikeSeq = Union[np.ndarray, Sequence[np.ndarray]]


# ----------------------------
# Core: experimental order q_k
# ----------------------------

def _as_2d_iterates(xk_seq):
    """
    Convert an input sequence of iterates into a 2D numpy array of shape (m, n).

    Accepted inputs:
      - np.ndarray of shape (m, n)
      - list/tuple of np.ndarray, each of shape (n,)
    """
    if isinstance(xk_seq, np.ndarray):
        x = np.asarray(xk_seq, dtype=float)
        if x.ndim == 1:
            # Single iterate (degenerate path)
            return x.reshape(1, -1)
        if x.ndim == 2:
            return x
        raise ValueError("xk_seq ndarray must be 1D or 2D.")
    else:
        # list/tuple of vectors
        xs = []
        for item in xk_seq:
            if item is None:
                continue
            v = np.asarray(item, dtype=float).reshape(-1)
            xs.append(v)
        if len(xs) == 0:
            return np.empty((0, 0), dtype=float)
        # ensure consistent dimension
        n = xs[0].shape[0]
        for i, v in enumerate(xs):
            if v.shape[0] != n:
                raise ValueError(f"Inconsistent dimensions in xk_seq at index {i}: {v.shape[0]} != {n}")
        return np.vstack(xs)


def experimental_orders_from_xk(xk_seq, norm = 2, eps = 1e-16,):
    """
    Compute the experimental order sequence q_k from iterates x_k, without x*.

    The returned array has length m (= number of iterates). Values that are not
    computable are set to NaN and the function never raises due to numerical issues.

    Index convention:
      - input iterates are x[0], x[1], ..., x[m-1]
      - differences: e[k] = x[k] - x[k-1] for k >= 1
      - q_k is computed for k = 2, ..., m-2 and stored at index k

    Numerical safeguards:
      - if any ||e|| <= eps -> NaN
      - if denominator log is too small -> NaN
      - any invalid log/ratio -> NaN
    """
    try:
        x = _as_2d_iterates(xk_seq)
    except Exception:
        # If input is malformed, degrade gracefully
        return np.array([np.nan], dtype=float)

    m = x.shape[0]
    q = np.full(m, np.nan, dtype=float)

    if m < 4:
        return q

    # e_norm[i] corresponds to ||e^(i+1)|| because e = x[1:] - x[:-1]
    e = x[1:] - x[:-1]  # shape (m-1, n)
    with np.errstate(over="ignore", invalid="ignore"):
        e_norm = np.linalg.norm(e, ord=norm, axis=1)

    # Compute q_k for k = 2..m-2
    # Mapping:
    #   ||e^(k+1)|| -> e_norm[k]     (since e_norm[0] = ||e^1||)
    #   ||e^(k)||   -> e_norm[k-1]
    #   ||e^(k-1)|| -> e_norm[k-2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for k in range(2, m - 1):
            if k > m - 2:
                break
            a = e_norm[k]      # ||e^(k+1)||
            b = e_norm[k - 1]  # ||e^(k)||
            c = e_norm[k - 2]  # ||e^(k-1)||

            # too small -> NaN
            if (not np.isfinite(a)) or (not np.isfinite(b)) or (not np.isfinite(c)):
                q[k] = np.nan
                continue
            if (a <= eps) or (b <= eps) or (c <= eps):
                q[k] = np.nan
                continue

            r1 = a / b
            r0 = b / c

            if (not np.isfinite(r1)) or (not np.isfinite(r0)) or (r1 <= 0.0) or (r0 <= 0.0):
                q[k] = np.nan
                continue

            num = np.log(r1)
            den = np.log(r0)

            # den too small -> unstable
            if (not np.isfinite(num)) or (not np.isfinite(den)) or (abs(den) <= eps):
                q[k] = np.nan
                continue

            q[k] = num / den

    return q


def experimental_rate_summary(q, tail = 5, agg = "median"):
    """
    Summarize the (estimated) order q from the tail of q_k.

    By default returns the median of the last `tail` finite values found in the tail window.
    If not enough finite values exist, returns NaN.
    """
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size == 0:
        return float("nan")

    tail = int(max(1, tail))
    tail_window = q_arr[-tail:]
    finite = tail_window[np.isfinite(tail_window)]
    if finite.size == 0:
        return float("nan")

    agg = str(agg).lower()
    if agg == "median":
        return float(np.median(finite))
    if agg == "mean":
        return float(np.mean(finite))
    if agg == "min":
        return float(np.min(finite))
    if agg == "max":
        return float(np.max(finite))

    raise ValueError("agg must be one of: 'median', 'mean', 'min', 'max'.")


# ----------------------------
# Plotting
# ----------------------------

def plot_convergence_orders(seqs, labels, title,save_path, max_points = None,
                            ylim = None, dpi = 220, figsize = (10.5, 6.0), legend_outside = True,
                            grid_alpha = 0.30,):
    """
    Plot multiple q_k sequences in a readable figure and save to disk.

    - seqs: list of q arrays
    - labels: list of curve labels
    - max_points: if set, plot only the LAST max_points points of each sequence
    - ylim: optional y-limits
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    for q, lab in zip(seqs, labels):
        q = np.asarray(q, dtype=float).reshape(-1)
        x = np.arange(q.size)

        if max_points is not None and max_points > 0 and q.size > max_points:
            q = q[-max_points:]
            x = x[-max_points:]

        ax.plot(x, q, label=lab, linewidth=1.6)

    ax.set_title(title)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("estimated order $q_k$")
    ax.grid(True, alpha=grid_alpha)

    if ylim is not None:
        ax.set_ylim(ylim)

    if len(labels) > 0:
        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", frameon=True)
        else:
            ax.legend(fontsize="small", frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
#Figure generation
# ----------------------------

def _norm_method(m):
    m = (m or "").strip().lower()
    if m in ("nm", "newton", "newtonmethod", "mod-newton", "modified_newton"):
        return "nm"
    if m in ("tr", "tn", "truncated", "truncatednewton", "truncatednewtonmethod", "trunc-newton"):
        return "tr"
    return m or "unknown"


def _norm_problem(p):
    return (p or "").strip().lower() or "unknown"


def _rates_filename(method, problem, n,deriv_type, mode = None, k_step = None):
    """
    Required naming:
      - rates_{method}_{problem}_n{n}_{exact|fd}.png
      - if fd: rates_{method}_{problem}_n{n}_fd_{mode}_k{k}.png
    """
    method = _norm_method(method)
    problem = _norm_problem(problem)
    n_str = f"n{int(n)}"

    deriv_type = (deriv_type or "").strip().lower()
    if deriv_type == "exact":
        return f"rates_{method}_{problem}_{n_str}_exact.png"
    if deriv_type == "fd":
        mode = (mode or "").strip().lower()
        k_step = int(k_step) if k_step is not None else None
        return f"rates_{method}_{problem}_{n_str}_fd_{mode}_k{k_step}.png"

    # fallback
    return f"rates_{method}_{problem}_{n_str}_{deriv_type}.png"


def _rates_title(method, problem, n, deriv_type, mode = None, k_step = None):
    method = _norm_method(method).upper()
    problem = _norm_problem(problem)
    deriv_type = (deriv_type or "").strip().lower()

    if deriv_type == "exact":
        return f"Experimental order $q_k$ | {method} | {problem} | n={int(n)} | exact"
    if deriv_type == "fd":
        return f"Experimental order $q_k$ | {method} | {problem} | n={int(n)} | FD | mode={mode} | k={k_step}"

    return f"Experimental order $q_k$ | {method} | {problem} | n={int(n)} | {deriv_type}"


def generate_convergence_rate_figures( runs,
    out_dir = "./figures/convergence_rates",
    tail = 5,
    agg = "median",
    max_points = 250,
    ylim = None,
    norm = 2,
    eps = 1e-16,
    include_q_in_label = True):
    """
    Generate and save all required figures from a list of run-record dicts.

    Expected run record fields (minimum):
      - method: 'nm' or 'tr'
      - problem: 'p31', 'p52', ...
      - n: dimension
      - type: 'exact' or 'fd'   (alias: 'deriv')
      - xk: iterate path (np.ndarray (m,n) or list of np.ndarray)
      - converged: bool

    For FD runs:
      - mode: 'scalar' or 'adaptive'
      - k: 4 / 8 / 12

    The function groups runs and creates:
      - exact plots: for each (problem, method, n)
      - fd plots: for each (problem, method, n, mode, k)
    """
    os.makedirs(out_dir, exist_ok=True)

    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)

    for r in runs:
        method = _norm_method(str(r.get("method", "")))
        problem = _norm_problem(str(r.get("problem", "")))
        n = int(r.get("n"))
        deriv_type = (r.get("type", r.get("deriv", "exact")) or "").strip().lower()

        if deriv_type == "fd":
            mode = (r.get("mode") or "").strip().lower()
            k_step = int(r.get("k"))
            key = (problem, method, n, "fd", mode, k_step)
        else:
            key = (problem, method, n, "exact", None, None)

        grouped[key].append(r)

    for (problem, method, n, deriv_type, mode, k_step), items in grouped.items():
        # keep only converged sequences
        items_conv = [it for it in items if bool(it.get("converged", False))]
        if len(items_conv) == 0:
            # nothing to plot
            continue

        seqs: List[np.ndarray] = []
        labels: List[str] = []

        for idx, it in enumerate(items_conv):
            xk = it.get("xk")
            qk = experimental_orders_from_xk(xk, norm=norm, eps=eps)
            seqs.append(qk)

            lab = str(it.get("label", f"run{idx+1}"))
            if include_q_in_label:
                q_est = experimental_rate_summary(qk, tail=tail, agg=agg)
                if np.isfinite(q_est):
                    lab = f"{lab} (q~{q_est:.3g})"
                else:
                    lab = f"{lab} (q~NaN)"
            labels.append(lab)

        title = _rates_title(method, problem, n, deriv_type, mode=mode, k_step=k_step)
        filename = _rates_filename(method, problem, n, deriv_type, mode=mode, k_step=k_step)
        save_path = os.path.join(out_dir, filename)

        plot_convergence_orders(
            seqs=seqs,
            labels=labels,
            title=title,
            save_path=save_path,
            max_points=max_points,
            ylim=ylim,
        )


# ----------------------------
# Convenience converters (optional)
# ----------------------------

def runs_from_iterate_tol_results(
    all_results,
    problem,
    tol = 1e-6):
    """
    Build run-records from parameters.iterate_tol(..., return_full=True) output.

    We expect all_results[tol] to contain:
      - 'x_initial'     : dict n -> {converges, path, ...}
      - 'x_random'      : dict n -> {paths: [path_i], converges_list: [bool_i], ...}
      - 'x_initial_tr'  : dict n -> {converges, path, ...}
      - 'x_random_tr'   : dict n -> {pathd or paths, converges_list, ...}

    If 'converges_list' is missing, we fall back to assuming all paths share the aggregated 'converges' flag.
    """
    if not all_results:
        return []

    # choose tol key
    if tol is None:
        # pick a "reasonable" default: 1e-6 if present, else the first key
        tol_key = 1e-6 if (1e-6 in all_results) else list(all_results.keys())[0]
    else:
        tol_key = tol if (tol in all_results) else list(all_results.keys())[0]

    block = all_results.get(tol_key, {})
    runs: List[Dict[str, Any]] = []

    # NM initial
    for n, met in (block.get("x_initial", {}) or {}).items():
        runs.append({
            "method": "nm",
            "problem": problem,
            "n": int(n),
            "type": "exact",
            "label": "initial",
            "converged": bool(met.get("converges", False)),
            "xk": met.get("path"),
        })

    # NM random
    for n, met in (block.get("x_random", {}) or {}).items():
        paths = met.get("paths", []) or []
        conv_list = met.get("converges_list", None)
        if conv_list is None:
            # fallback: replicate the aggregated flag
            conv_list = [bool(met.get("converges", False))] * len(paths)

        for i, pth in enumerate(paths):
            runs.append({
                "method": "nm",
                "problem": problem,
                "n": int(n),
                "type": "exact",
                "label": f"rand{i+1}",
                "converged": bool(conv_list[i]) if i < len(conv_list) else bool(met.get("converges", False)),
                "xk": pth,
            })

    # TR initial
    for n, met in (block.get("x_initial_tr", {}) or {}).items():
        runs.append({
            "method": "tr",
            "problem": problem,
            "n": int(n),
            "type": "exact",
            "label": "initial",
            "converged": bool(met.get("converges", False)),
            "xk": met.get("path"),
        })

    # TR random
    for n, met in (block.get("x_random_tr", {}) or {}).items():
        # some drivers used 'pathd' for TR
        paths = met.get("paths", None)
        if paths is None:
            paths = met.get("pathd", []) or []

        conv_list = met.get("converges_list", None)
        if conv_list is None:
            conv_list = [bool(met.get("converges", False))] * len(paths)

        for i, pth in enumerate(paths):
            runs.append({
                "method": "tr",
                "problem": problem,
                "n": int(n),
                "type": "exact",
                "label": f"rand{i+1}",
                "converged": bool(conv_list[i]) if i < len(conv_list) else bool(met.get("converges", False)),
                "xk": pth,
            })

    return runs


def runs_from_iterate_fd_tables(x_initial_fd, x_random_fd,
    problem,
    method =  "nm",
):
    """
    Build run-records from parameters.iterate_fd outputs.

    This function is intentionally tolerant:
      - x_initial_fd and x_random_fd can be pandas DataFrames, or list-of-dicts.
      - It looks for fields: n, mode, k, converges, path / paths, converges_list.

    If converges_list is missing in the random table, it falls back to the aggregated 'converges' flag.
    """
    def to_records(obj: Any) -> List[Dict[str, Any]]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        # pandas DataFrame has to_dict
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict(orient="records")
            except TypeError:
                # some objects use different signature
                pass
        # fallback: try iterrows-like
        if hasattr(obj, "iterrows"):
            recs = []
            for _, row in obj.iterrows():
                recs.append(dict(row))
            return recs
        return []

    init_recs = to_records(x_initial_fd)
    rand_recs = to_records(x_random_fd)

    runs: List[Dict[str, Any]] = []

    # initial: each row corresponds to a single starting point (usually the unique x0 for that n)
    for r in init_recs:
        runs.append({
            "method": method,
            "problem": problem,
            "n": int(r.get("n")),
            "type": "fd",
            "mode": str(r.get("mode", "")).lower(),
            "k": int(r.get("k")),
            "label": "initial",
            "converged": bool(r.get("converges", False)),
            "xk": r.get("path"),
        })

    # random: each row contains paths for multiple starting points
    for r in rand_recs:
        paths = r.get("paths", []) or []
        conv_list = r.get("converges_list", None)
        if conv_list is None:
            conv_list = [bool(r.get("converges", False))] * len(paths)

        for i, pth in enumerate(paths):
            runs.append({
                "method": method,
                "problem": problem,
                "n": int(r.get("n")),
                "type": "fd",
                "mode": str(r.get("mode", "")).lower(),
                "k": int(r.get("k")),
                "label": f"rand{i+1}",
                "converged": bool(conv_list[i]) if i < len(conv_list) else bool(r.get("converges", False)),
                "xk": pth,
            })

    return runs
