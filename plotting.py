import csv
import re
import sys
import importlib
from pathlib import Path

import numpy as np

# Headless backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import matplotlib.patheffects as patheffects


# ----------------------------
# Config
# ----------------------------

CSV_ROOT = Path("csv")
OUT_ROOT = Path("figures") / "plots"

# New single paths table (n=2 only)
PATHS_CSV = CSV_ROOT / "path" / "paths_n2.csv"

GRID_RES = 220              # grid resolution per axis for background (keep moderate for speed)
GRID_LEVELS = 120           # more levels => smoother fill
BG_GAMMA = 0.85             # <1 => smoother contrast
VIRIDIS_BLEND = 0.18        # 0..1: blend viridis towards white (less "acceso")

DRAW_CONTOUR_LINES = True
CONTOUR_LINE_LEVELS = 14

PADDING_FRAC = 0.15
MIN_PADDING_ABS = 0.5
MAX_RANDOM_PATHS = 5

FIGSIZE = (10.5, 7.2)
DPI = 220

TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 13
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12

# Path styling
INIT_LINEWIDTH = 2.0
RAND_LINEWIDTH = 1.8
INIT_MARKERSIZE = 6.0
RAND_MARKERSIZE = 5.6
MARKER_EDGEWIDTH = 0.7
STROKE_EXTRA = 1.2          # extra linewidth for black stroke
STROKE_EXTRA_RAND = 1.1

# Use tab10 for path colors
TAB10 = plt.get_cmap("tab10")


# ----------------------------
# Logging
# ----------------------------

def log_info(msg):
    print(f"[INFO] {msg}")

def log_skip(msg):
    print(f"[SKIP] {msg}")

def log_warn(msg):
    print(f"[WARN] {msg}")

def log_err(msg):
    print(f"[ERROR] {msg}")


# ----------------------------
# Colormap: soft viridis
# ----------------------------

def make_soft_viridis(blend=0.18):
    base = plt.get_cmap("viridis")
    cols = base(np.linspace(0, 1, 256))
    cols[:, :3] = (1.0 - blend) * cols[:, :3] + blend * 1.0
    return LinearSegmentedColormap.from_list("viridis_soft", cols)

SOFT_VIRIDIS = make_soft_viridis(VIRIDIS_BLEND)


# ----------------------------
# Dynamic import + objective wrapper
# ----------------------------

def import_problem(problem_id: int):
    """
    Dynamically import:
      from Problems.Problem_<ID> import Problem_<ID>
    """
    module_name = f"Problems.Problem_{problem_id}"
    class_name = f"Problem_{problem_id}"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}") from e

    if not hasattr(mod, class_name):
        raise ImportError(f"Module '{module_name}' does not define class '{class_name}'")

    return getattr(mod, class_name)

def eval_objective(problem, x):
    """
    Wrapper to evaluate F(x) trying (in order): function, F, objective.
    """
    for name in ("function", "F", "objective"):
        if hasattr(problem, name):
            attr = getattr(problem, name)
            if callable(attr):
                return float(attr(x))
    raise AttributeError(
        "Problem instance exposes none of callable methods: function(x), F(x), objective(x)"
    )


# ----------------------------
# Labels mapping
# ----------------------------

def method_label(method):
    m = (method or "").strip().lower()
    if m == "nm":
        return "Mod. N. M."
    if m == "tr":
        return "Tr. N. M."
    return (method or "").strip().upper()

def case_label(case_name):
    c = (case_name or "").strip().lower()
    mapping = {
        "exact": "Exact",
        "mixed fd": "F.D. Hess.",
        "full fd": "Full F.D.",
    }
    return mapping.get(c, case_name)

def fd_cfg_label(case_name, h_mode, k_fd):
    """
    Extra label only when FD config is meaningful.
    """
    c = (case_name or "").strip().lower()
    hm = (h_mode or "").strip().lower()
    k = None if k_fd is None else int(k_fd)

    if c in ("mixed fd", "full fd"):
        if hm and k is not None:
            return f"h={hm}, k={k}"
        if hm:
            return f"h={hm}"
        if k is not None:
            return f"k={k}"
    return ""

# ----------------------------
# Robust CSV parsing helpers
# ----------------------------

def _normalize_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "").replace("\u00a0", " ").replace("\u200b", "")
    return s.strip()

def _find_colname(fieldnames, candidates):
    lower_map = {fn.strip().lower(): fn for fn in fieldnames if fn is not None}
    for c in candidates:
        key = c.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None

_PROB_ID_RE = re.compile(r"(\d+)")

def _parse_problem_id(problem_field: str):
    """
    Accepts 'p52', 'Problem_52', '52', etc.
    Returns int or None.
    """
    s = _normalize_text(problem_field).lower()
    if not s:
        return None
    m = _PROB_ID_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _parse_int_maybe(s):
    s = _normalize_text(s)
    if s == "" or s.lower() in ("nan", "none", "-"):
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def _parse_float_maybe(s):
    s = _normalize_text(s)
    if s == "" or s.lower() in ("nan", "none", "-"):
        return None
    try:
        return float(s)
    except Exception:
        return None


# ----------------------------
# Scanning & grouping (NEW: single CSV)
# ----------------------------

def scan_groups_from_paths_csv(paths_csv: Path):
    """
    Reads csv/path/paths_n2.csv and creates plot groups.

    A "plot group" corresponds to ONE figure:
      (problem_id, method, case, h_mode, k_fd, tol)

    Inside each group we reconstruct:
      - initial path (xbar)
      - up to MAX_RANDOM_PATHS random paths

    Returns list of group dicts:
      {
        problem_id, method, case_name, h_mode, k_fd, tol,
        initial_path: np.ndarray (m,2),
        random_paths: [np.ndarray (mi,2), ...],
      }
    """
    paths_csv = Path(paths_csv)
    if not paths_csv.exists():
        log_skip(f"Paths CSV not found: '{paths_csv}'.")
        return []

    # allow huge files safely
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    # plot_key -> dict of path_key -> list[(k,x1,x2)]
    plot_buckets = {}

    with paths_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            log_err("paths_n2.csv has no header.")
            return []

        # required columns (case-insensitive)
        col_problem = _find_colname(reader.fieldnames, ["problem"])
        col_method = _find_colname(reader.fieldnames, ["method"])
        col_case = _find_colname(reader.fieldnames, ["case"])
        col_h_mode = _find_colname(reader.fieldnames, ["h_mode", "hmode"])
        col_k_fd = _find_colname(reader.fieldnames, ["k_fd", "kfd"])
        col_tol = _find_colname(reader.fieldnames, ["tol"])
        col_start_type = _find_colname(reader.fieldnames, ["start_type", "starttype"])
        col_path_id = _find_colname(reader.fieldnames, ["path_id", "pathid"])
        col_run_id = _find_colname(reader.fieldnames, ["run_id", "runid"])
        col_k = _find_colname(reader.fieldnames, ["k"])
        col_x1 = _find_colname(reader.fieldnames, ["x1"])
        col_x2 = _find_colname(reader.fieldnames, ["x2"])

        missing = []
        for name, col in [
            ("Problem", col_problem),
            ("Method", col_method),
            ("Case", col_case),
            ("start_type", col_start_type),
            ("path_id", col_path_id),
            ("run_id", col_run_id),
            ("k", col_k),
            ("x1", col_x1),
            ("x2", col_x2),
        ]:
            if not col:
                missing.append(name)

        if missing:
            log_err(f"paths_n2.csv missing required columns: {missing}")
            return []

        # h_mode / k_fd / tol are optional for Exact; but if missing entirely, we just treat as blank
        # (still groupable).
        for row_idx, row in enumerate(reader, start=1):
            problem_field = row.get(col_problem, "")
            pid = _parse_problem_id(problem_field)
            if pid is None:
                continue

            method = _normalize_text(row.get(col_method, "")).lower()
            case = _normalize_text(row.get(col_case, ""))

            h_mode = _normalize_text(row.get(col_h_mode, "")) if col_h_mode else ""
            k_fd = _parse_int_maybe(row.get(col_k_fd, "")) if col_k_fd else None
            tol = _normalize_text(row.get(col_tol, "")) if col_tol else ""

            start_type = _normalize_text(row.get(col_start_type, "")).lower()
            path_id = _normalize_text(row.get(col_path_id, ""))
            run_id = _parse_int_maybe(row.get(col_run_id, ""))

            kk = _parse_int_maybe(row.get(col_k, ""))
            x1 = _parse_float_maybe(row.get(col_x1, ""))
            x2 = _parse_float_maybe(row.get(col_x2, ""))

            if method == "" or case == "" or start_type == "" or path_id == "" or run_id is None:
                continue
            if kk is None or x1 is None or x2 is None:
                continue

            # normalize: Exact may have empty h_mode/k_fd -> keep as ""/None
            h_mode_norm = (h_mode or "").strip().lower()
            k_fd_norm = k_fd  # int or None
            tol_norm = (tol or "").strip()

            plot_key = (pid, method, case, h_mode_norm, k_fd_norm, tol_norm)
            path_key = (start_type, path_id, int(run_id))

            if plot_key not in plot_buckets:
                plot_buckets[plot_key] = {}
            if path_key not in plot_buckets[plot_key]:
                plot_buckets[plot_key][path_key] = []
            plot_buckets[plot_key][path_key].append((int(kk), float(x1), float(x2)))

    groups = []
    for (pid, method, case, h_mode, k_fd, tol), paths_dict in plot_buckets.items():
        # rebuild paths as arrays sorted by iteration k
        rebuilt = []
        for (start_type, path_id, run_id), triplets in paths_dict.items():
            triplets.sort(key=lambda t: t[0])  # sort by k
            arr = np.array([(t[1], t[2]) for t in triplets], dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
                continue
            rebuilt.append((start_type, path_id, run_id, arr))

        if not rebuilt:
            continue

        # pick xbar (initial) path
        initial_candidates = [(st, pid_, rid, arr) for (st, pid_, rid, arr) in rebuilt
                              if st == "initial" or pid_.strip().lower() == "xbar"]
        if not initial_candidates:
            # if missing, still allow plot but we'll skip (require xbar to keep same expected layout)
            log_skip(f"Missing initial/xbar path for p{pid} {case} {method} (h_mode={h_mode}, k_fd={k_fd}).")
            continue
        initial_candidates.sort(key=lambda t: (t[2], t[1]))  # by run_id then path_id
        initial_path = initial_candidates[0][3]

        # collect random paths
        random_candidates = [(st, pid_, rid, arr) for (st, pid_, rid, arr) in rebuilt if st == "random"]

        def _rand_sort_key(t):
            # t: (st, path_id, run_id, arr)
            s = t[1].strip().lower()
            m = re.search(r"(\d+)", s)
            num = int(m.group(1)) if m else 10**9
            return (num, t[2], s)

        random_candidates.sort(key=_rand_sort_key)
        random_paths = [t[3] for t in random_candidates[:MAX_RANDOM_PATHS]]

        if not random_paths:
            log_skip(f"No random paths for p{pid} {case} {method} (h_mode={h_mode}, k_fd={k_fd}).")
            continue

        groups.append({
            "problem_id": pid,
            "method": method,
            "case_name": case,
            "h_mode": h_mode,
            "k_fd": k_fd,
            "tol": tol,
            "initial_path": initial_path,
            "random_paths": random_paths,
        })

    groups.sort(key=lambda g: (g["problem_id"], str(g["case_name"]), str(g["method"]), str(g["h_mode"]), g["k_fd"] if g["k_fd"] is not None else -1))
    return groups


# ----------------------------
# Plot helpers
# ----------------------------

def _collect_all_points(paths):
    if not paths:
        return np.empty((0, 2), dtype=float)
    good = []
    for p in paths:
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2 and p.shape[0] > 0:
            good.append(p)
    if not good:
        return np.empty((0, 2), dtype=float)
    return np.vstack(good)

def _compute_bbox(points):
    x1 = points[:, 0]
    x2 = points[:, 1]
    xmin, xmax = float(np.min(x1)), float(np.max(x1))
    ymin, ymax = float(np.min(x2)), float(np.max(x2))

    dx = xmax - xmin
    dy = ymax - ymin

    pad_x = max(MIN_PADDING_ABS, PADDING_FRAC * (dx if dx > 1e-12 else 1.0))
    pad_y = max(MIN_PADDING_ABS, PADDING_FRAC * (dy if dy > 1e-12 else 1.0))

    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y

def _eval_grid(problem, xmin, xmax, ymin, ymax):
    xs = np.linspace(xmin, xmax, GRID_RES)
    ys = np.linspace(ymin, ymax, GRID_RES)
    X, Y = np.meshgrid(xs, ys)

    xy = np.column_stack([X.ravel(), Y.ravel()])
    z_flat = np.empty(xy.shape[0], dtype=float)

    with np.errstate(all="ignore"):
        for i in range(xy.shape[0]):
            try:
                z_flat[i] = eval_objective(problem, xy[i])
            except Exception:
                z_flat[i] = np.nan

    Z = z_flat.reshape(X.shape)

    finite = np.isfinite(Z)
    if not np.any(finite):
        raise ValueError("Objective evaluation on grid produced no finite values.")

    z_min = float(np.nanmin(Z[finite]))
    z_max = float(np.nanmax(Z[finite]))

    # Replace NaN/Inf with max for contourf stability
    Z = np.where(np.isfinite(Z), Z, z_max)

    # Flat surface guard
    if abs(z_max - z_min) < 1e-14:
        Z = Z + 1e-12 * np.random.standard_normal(Z.shape)

    return X, Y, Z


def _slug(s: str):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    return s


# ----------------------------
# Plotting core
# ----------------------------

def plot_group(group):
    pid = group["problem_id"]
    case = group["case_name"]
    method = group["method"]
    h_mode = group.get("h_mode", "") or ""
    k_fd = group.get("k_fd", None)
    tol = group.get("tol", "") or ""

    initial_path = group["initial_path"]
    random_paths = group["random_paths"]

    # Import problem and instantiate n=2
    try:
        ProblemCls = import_problem(pid)
        problem = ProblemCls(2)
    except Exception as e:
        log_skip(f"Problem import failed for ID={pid}: {e}")
        return False

    # Compute bbox from all points
    all_points = _collect_all_points([initial_path] + list(random_paths))
    if all_points.shape[0] == 0:
        log_skip(f"No points to plot for p{pid} {case} {method}.")
        return False

    xmin, xmax, ymin, ymax = _compute_bbox(all_points)

    # Evaluate objective on grid
    try:
        X, Y, Z = _eval_grid(problem, xmin, xmax, ymin, ymax)
    except Exception as e:
        log_skip(f"Grid objective evaluation failed for p{pid} {case} {method}: {e}")
        return False

    # Prepare output path (same folder structure as before)
    out_dir = OUT_ROOT / f"p{pid}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # File naming: keep old prefix, add FD qualifiers when needed
    base = f"Plot_P{pid}_{_slug(case)}_{_slug(method)}"
    fd_extra = fd_cfg_label(case, h_mode, k_fd)
    if fd_extra:
        base += f"_{_slug(h_mode)}_k{int(k_fd) if k_fd is not None else ''}"
    # Optionally include tol if you ever run multiple tolerances (kept off by default)
    # if tol:
    #     base += f"_tol{_slug(tol)}"

    out_file = out_dir / f"{base}.png"

    # --- Plot ---
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.gca()

    # Background: soft viridis + smooth contrast
    ax.contourf(
        X, Y, Z,
        levels=GRID_LEVELS,
        cmap=SOFT_VIRIDIS,
        norm=PowerNorm(gamma=BG_GAMMA),
        antialiased=True
    )

    # Optional: contour lines overlay + labels
    if DRAW_CONTOUR_LINES:
        try:
            cs = ax.contour(X, Y, Z, levels=CONTOUR_LINE_LEVELS, alpha=0.25, linewidths=0.6)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
        except Exception:
            pass

    # Overlay paths with black outlines
    init_color = TAB10(0)
    line_init, = ax.plot(
        initial_path[:, 0], initial_path[:, 1],
        "-o",
        color=init_color,
        markerfacecolor=init_color,
        markeredgecolor="black",
        markeredgewidth=MARKER_EDGEWIDTH,
        linewidth=INIT_LINEWIDTH,
        markersize=INIT_MARKERSIZE,
        label="xbar",
        zorder=5,
    )
    line_init.set_path_effects([
        patheffects.Stroke(linewidth=INIT_LINEWIDTH + STROKE_EXTRA, foreground="black"),
        patheffects.Normal(),
    ])

    for i, p in enumerate(random_paths[:MAX_RANDOM_PATHS], start=1):
        c = TAB10(i % 10)
        line_r, = ax.plot(
            p[:, 0], p[:, 1],
            "-o",
            color=c,
            markerfacecolor=c,
            markeredgecolor="black",
            markeredgewidth=MARKER_EDGEWIDTH,
            linewidth=RAND_LINEWIDTH,
            markersize=RAND_MARKERSIZE,
            label=f"Random {i}",
            zorder=6,
        )
        line_r.set_path_effects([
            patheffects.Stroke(linewidth=RAND_LINEWIDTH + STROKE_EXTRA_RAND, foreground="black"),
            patheffects.Normal(),
        ])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$x_1$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$x_2$", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    # Title: remove "(n=2)" as requested, add FD details when relevant
    title = f"Problem {pid} - {method_label(method)} - {case_label(case)}"
    fd_extra = fd_cfg_label(case, h_mode, k_fd)
    if fd_extra:
        title += f" - {fd_extra}"
    # Optional: add tol if you want it explicit in the title (disabled by default)
    # if tol:
    #     title += f" - tol={tol}"

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, frameon=True)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    log_info(f"Saved: {out_file}")
    return True


def main():
    log_info(f"Reading paths table: {PATHS_CSV}")
    groups = scan_groups_from_paths_csv(PATHS_CSV)

    if not groups:
        log_skip("No valid plot groups found. Check csv/path/paths_n2.csv content.")
        return 0

    saved = 0
    for g in groups:
        ok = plot_group(g)
        if ok:
            saved += 1

    log_info(f"Done. Figures saved: {saved}/{len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
