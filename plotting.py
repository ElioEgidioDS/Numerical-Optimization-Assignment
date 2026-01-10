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

# Float regex (supports scientific notation)
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

# Folder pattern: p31_final -> 31
PROBLEM_DIR_RE = re.compile(r"^p(\d+)_final$")

# File pattern: x_initial_<method>.csv / x_random_<method>.csv
FILE_RE = re.compile(r"^x_(initial|random)_(.+)\.csv$", re.IGNORECASE)


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

def import_problem(problem_id):
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
        "final_1": "Exact",
        "final_2": "F.D. Hess.",
        "final_3": "Full F.D.",
    }
    return mapping.get(c, case_name)


# ----------------------------
# Robust CSV parsing
# ----------------------------

def _normalize_text(s):
    """
    Normalize BOM/NBSP/odd spaces and tame some numpy-ish tokens that would pollute float extraction.
    """
    if s is None:
        return ""

    s = str(s)

    # Remove BOM and NBSP and zero-width spaces
    s = s.replace("\ufeff", "").replace("\u00a0", " ").replace("\u200b", "")

    # Remove dtype / type tokens that include digits (float64 -> float, int32 -> int, etc.)
    # This prevents capturing the "64"/"32" as numbers.
    s = re.sub(r"\b(float|int)\d+\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(np|numpy)\.(float|int)\d+\b", r"\1.\2", s, flags=re.IGNORECASE)

    # Remove common dtype=... snippets
    s = re.sub(r"dtype\s*=\s*[^,\]\)]+", "dtype", s, flags=re.IGNORECASE)

    # Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    return s.strip()

def _match_brackets(s, start_idx):
    """
    Given s and index of an opening '[', find the matching closing ']' using nesting count.
    Returns end index (inclusive) or -1 if not found.
    """
    if start_idx < 0 or start_idx >= len(s) or s[start_idx] != "[":
        return -1
    depth = 0
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1

def _extract_outer_bracket_block(s):
    """
    Extract the outermost bracketed block from first '[' to its matching ']'.
    """
    lb = s.find("[")
    if lb < 0:
        return None
    rb = _match_brackets(s, lb)
    if rb < 0:
        return None
    return s[lb:rb + 1]

def _extract_array_blocks(s):
    """
    If string contains 'array(...[[...]]...)', extract each bracket block belonging to each array.
    """
    blocks = []
    idx = 0
    while True:
        pos = s.find("array", idx)
        if pos < 0:
            break
        lb = s.find("[", pos)
        if lb < 0:
            idx = pos + 4
            continue
        rb = _match_brackets(s, lb)
        if rb < 0:
            idx = lb + 1
            continue
        blocks.append(s[lb:rb + 1])
        idx = rb + 1
    return blocks

def _block_to_path_2d(block):
    """
    Convert a bracketed string '[[...]]' into an array (m,2) by float-regex + reshape.
    """
    block = _normalize_text(block)

    nums = FLOAT_RE.findall(block)
    if not nums:
        raise ValueError("No numeric values found in block.")

    arr = np.array([float(v) for v in nums], dtype=float)
    if arr.size % 2 != 0:
        raise ValueError(f"Found {arr.size} floats, not divisible by 2 -> cannot reshape to (-1,2).")

    path = arr.reshape(-1, 2)
    if path.shape[0] < 1:
        raise ValueError("Parsed path has zero points.")

    return path

def _parse_field_into_paths(field):
    """
    Parse one CSV field into 1 or multiple 2D paths.
    Handles:
      - plain '[[...]]' (multiline, missing commas)
      - '[array([[...]]), array([[...]]), ...]'
    """
    field = _normalize_text(field)
    if not field:
        return []

    paths = []

    if "array" in field:
        blocks = _extract_array_blocks(field)
        if not blocks:
            outer = _extract_outer_bracket_block(field)
            if outer:
                blocks = [outer]
        for b in blocks:
            try:
                paths.append(_block_to_path_2d(b))
            except Exception as e:
                log_warn(f"Failed parsing one array-block: {e}")
        return paths

    outer = _extract_outer_bracket_block(field)
    if not outer:
        return []
    try:
        paths.append(_block_to_path_2d(outer))
    except Exception as e:
        log_warn(f"Failed parsing outer bracket block: {e}")
    return paths

def _find_colname(fieldnames, candidates):
    """
    Case-insensitive match of any candidate in fieldnames.
    Returns the actual fieldname as it appears in CSV.
    """
    lower_map = {fn.strip().lower(): fn for fn in fieldnames if fn is not None}
    for c in candidates:
        key = c.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None

def parse_csv_paths_for_n2(csv_path, max_paths=5):
    """
    Read a CSV with csv.DictReader, filter rows with n==2, parse a field ('path'/'paths')
    into 2D paths. Returns up to max_paths paths.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    # Allow huge multiline fields
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    out_paths = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header/fieldnames.")

        n_col = _find_colname(reader.fieldnames, ["n"])
        if not n_col:
            raise ValueError("CSV missing required column 'n' (needed to filter n==2).")

        path_col = _find_colname(reader.fieldnames, ["path", "paths", "pathd"])
        if not path_col:
            raise ValueError("CSV missing path column among: 'path', 'paths' (also accepted: 'pathd').")

        for row_idx, row in enumerate(reader, start=1):
            raw_n = row.get(n_col, "")
            try:
                n_val = int(float(str(raw_n).strip()))
            except Exception:
                continue

            if n_val != 2:
                continue

            field = row.get(path_col, "")
            if field is None:
                continue

            parsed = _parse_field_into_paths(field)
            for p in parsed:
                if not (isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2):
                    log_warn(f"Row {row_idx}: parsed path has shape {getattr(p, 'shape', None)}, expected (m,2). Skipping.")
                    continue
                out_paths.append(p)

            if len(out_paths) >= max_paths:
                break

    return out_paths[:max_paths]


# ----------------------------
# Scanning and grouping
# ----------------------------

def scan_groups(csv_root):
    """
    Scan csv_root for folders like p31_final/final_1/ containing x_initial_<method>.csv and x_random_<method>.csv.
    Build valid groups only if both initial and random exist for same method.
    Returns list of dicts: {problem_id, case_name, method, initial_csv, random_csv}.
    """
    csv_root = Path(csv_root)
    groups = {}  # key=(pid, case, method) -> {"initial": path, "random": path}

    if not csv_root.exists():
        log_skip(f"Input folder '{csv_root}' not found.")
        return []

    for prob_dir in csv_root.iterdir():
        if not prob_dir.is_dir():
            continue
        m = PROBLEM_DIR_RE.match(prob_dir.name)
        if not m:
            continue
        problem_id = int(m.group(1))

        for case_dir in prob_dir.iterdir():
            if not case_dir.is_dir():
                continue
            case_name = case_dir.name

            for f in case_dir.iterdir():
                if not f.is_file() or f.suffix.lower() != ".csv":
                    continue
                fm = FILE_RE.match(f.name)
                if not fm:
                    continue
                kind = fm.group(1).lower()     # initial/random
                method = fm.group(2).strip()   # nm / tr / ...

                key = (problem_id, case_name, method)
                if key not in groups:
                    groups[key] = {}
                groups[key][kind] = f

    out = []
    for (pid, case, method), d in groups.items():
        if "initial" in d and "random" in d:
            out.append({
                "problem_id": pid,
                "case_name": case,
                "method": method,
                "initial_csv": d["initial"],
                "random_csv": d["random"],
            })
        else:
            missing = []
            if "initial" not in d:
                missing.append("initial")
            if "random" not in d:
                missing.append("random")
            log_skip(f"Incomplete group p{pid} {case} {method}: missing {missing}")

    out.sort(key=lambda g: (g["problem_id"], g["case_name"], g["method"]))
    return out


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


# ----------------------------
# Plotting core
# ----------------------------

def plot_group(group):
    pid = group["problem_id"]
    case = group["case_name"]
    method = group["method"]

    # Import problem and instantiate n=2
    try:
        ProblemCls = import_problem(pid)
        problem = ProblemCls(2)
    except Exception as e:
        log_skip(f"Problem import failed for ID={pid}: {e}")
        return False

    # Parse initial (expect 1 path)
    try:
        initial_paths = parse_csv_paths_for_n2(group["initial_csv"], max_paths=1)
    except Exception as e:
        log_skip(f"Failed parsing initial CSV '{group['initial_csv']}': {e}")
        return False

    if not initial_paths:
        log_skip(f"No n==2 initial path found in '{group['initial_csv']}'.")
        return False

    initial_path = initial_paths[0]

    # Parse random (up to 5 paths)
    try:
        random_paths = parse_csv_paths_for_n2(group["random_csv"], max_paths=MAX_RANDOM_PATHS)
    except Exception as e:
        log_skip(f"Failed parsing random CSV '{group['random_csv']}': {e}")
        return False

    if not random_paths:
        log_skip(f"No n==2 random paths found in '{group['random_csv']}'.")
        return False

    # Compute bbox from all points
    all_points = _collect_all_points([initial_path] + random_paths)
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

    # Prepare output path
    out_dir = OUT_ROOT / f"p{pid}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Plot_P{pid}_{case}_{method}.png"

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

    title = f"Problem {pid} - {method_label(method)} - {case_label(case)} (n=2)"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, frameon=True)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    log_info(f"Saved: {out_file}")
    return True


def main():
    log_info("Scanning CSV folders...")
    groups = scan_groups(CSV_ROOT)

    if not groups:
        log_skip("No valid groups found. Check csv/ structure and filenames.")
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
