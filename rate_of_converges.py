import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
# needed to save plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# handles plotting of experimental convergence rates by reading the csv
# filter out runs that fail to get significant plots


DEFAULT_CSV_PATH = Path("csv") / "final" / "final_results.csv"
DEFAULT_OUT_ROOT = Path("figures") / "convergence_rates"

# outlier handling
MIN_Q = 0.0
MAX_Q = 10.0

# color definition
METHOD_COLORS = {
    "nm": "#1f77b4",   # blue
    "tr": "#7f7f7f",   # gray
}


RNG = np.random.default_rng(352283)

# plot style
TITLE_FONTSIZE = 20
SUBTITLE_FONTSIZE = 17
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

FIG_DPI = 260

# builds name
def filename(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s

# helpers
def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
def to_bool_success(x):
    return str(x).strip().lower() == "true"
def numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def method_sort_key(m):
    order = {"nm": 0, "tr": 1}
    m2 = str(m).strip().lower()
    return (order.get(m2, 99), m2)

# plots formatting
def pretty_case(case):
    return re.sub(r"\s+", " ", str(case).strip())
def method_color(m):
    m2 = str(m).strip().lower()
    return METHOD_COLORS.get(m2, "#333333")
def add_reference_lines(ax):
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.4, alpha=0.9, zorder=1)
    ax.axhline(2.0, color="red", linestyle="--", linewidth=1.4, alpha=0.9, zorder=1)
def set_axes_style(ax):
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.25)
def make_legend_handles():
    handles = [
        Line2D(
            [0], [0],
            marker="X",
            color="black",
            linestyle="None",
            markersize=11,
            markeredgewidth=1.0,
            label="xbar",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="black",
            linestyle="None",
            markersize=9,
            markeredgewidth=1.0,
            label="random starts",
        ),
        Patch(
            facecolor=method_color("nm"),
            edgecolor="black",
            alpha=0.40,
            label="nm",
        ),
        Patch(
            facecolor=method_color("tr"),
            edgecolor="black",
            alpha=0.40,
            label="tr",
        ),
    ]
    return handles
def compute_ylim_from_values(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 2.5)
    ymax = float(np.nanmax(v))
    ymax = max(2.5, ymax * 1.15)
    return (0.0, ymax)


# defines plots for exact derivatives
def plot_exact_case(df, title, outpath_png):
    if df.empty:
        return

    methods = sorted(df["Method"].unique().tolist(), key=method_sort_key)
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(methods), dtype=float)

    # bar means
    means = []
    for m in methods:
        vals = df.loc[df["Method"] == m, "ConvergenceRate"].astype(float).to_numpy()
        means.append(np.nanmean(vals) if vals.size else np.nan)


    bar_colors = [method_color(m) for m in methods]
    ax.bar(
        x,
        means,
        width=0.55,
        color=bar_colors,
        alpha=0.40,
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )

    # points
    for i, m in enumerate(methods):
        d_m = df[df["Method"] == m].copy()
        if d_m.empty:
            continue

        y = d_m["ConvergenceRate"].astype(float).to_numpy()
        jitter = RNG.normal(loc=0.0, scale=0.06, size=y.size)
        xs = (i + jitter).astype(float)

        c = method_color(m)
        is_xbar = d_m["point_id"].astype(str).str.lower() == "xbar"

        if is_xbar.any():
            ax.scatter(
                xs[is_xbar.to_numpy()],
                y[is_xbar.to_numpy()],
                s=110,
                marker="X",
                color=c,
                edgecolors="black",
                linewidths=1.0,
                zorder=3,
            )

        if (~is_xbar).any():
            ax.scatter(
                xs[(~is_xbar).to_numpy()],
                y[(~is_xbar).to_numpy()],
                s=75,
                marker="o",
                color=c,
                edgecolors="black",
                linewidths=0.9,
                zorder=3,
            )

    add_reference_lines(ax)
    set_axes_style(ax)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=14)
    ax.set_xlabel("method", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("convergence rate q", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x, methods)

    y0, y1 = compute_ylim_from_values(df["ConvergenceRate"].astype(float).to_numpy())
    ax.set_ylim(bottom=y0, top=y1)


    handles = make_legend_handles()
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    ensure_dir(outpath_png.parent)
    fig.savefig(outpath_png, dpi=FIG_DPI)
    plt.close(fig)


# defines plots for fd derivatives
def plot_fd_case(df, title, outpath_png):

    if df.empty:
        return

    if "k_fd" not in df.columns or "h_mode" not in df.columns:
        return

    methods = sorted(df["Method"].unique().tolist(), key=method_sort_key)
    if not methods:
        return

    df = df.copy()
    df["k_num"] = numeric_series(df["k_fd"])
    df = df[np.isfinite(df["k_num"].to_numpy())]
    if df.empty:
        return

    h_vals = df["h_mode"].dropna().astype(str).str.strip().unique().tolist()
    h_modes = sorted(h_vals) if h_vals else [""]

    n_panels = len(h_modes)
    fig_w = 15 if n_panels > 1 else 11
    fig_h = 7
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    # global y-limits from all values in this figure (already outlier-filtered in main)
    y0, y1 = compute_ylim_from_values(df["ConvergenceRate"].astype(float).to_numpy())

    for ax, hm in zip(axes, h_modes):
        d_h = df[df["h_mode"].astype(str).str.strip() == str(hm).strip()].copy()
        if d_h.empty:
            ax.set_axis_off()
            continue

        k_vals = sorted(d_h["k_num"].unique().tolist())
        base = np.arange(len(k_vals), dtype=float)

        group_width = 0.82
        bar_w = group_width / max(1, len(methods))

        for j, m in enumerate(methods):
            c = method_color(m)
            offset = (j - (len(methods) - 1) / 2.0) * bar_w

            means = []
            for kv in k_vals:
                vals = d_h[(d_h["Method"] == m) & (d_h["k_num"] == kv)]["ConvergenceRate"].astype(float).to_numpy()
                means.append(np.nanmean(vals) if vals.size else np.nan)

            ax.bar(
                base + offset,
                means,
                width=bar_w * 0.96,
                color=c,
                alpha=0.40,
                edgecolor="black",
                linewidth=1.2,
                zorder=2,
            )

            # points
            for i_k, kv in enumerate(k_vals):
                d_km = d_h[(d_h["Method"] == m) & (d_h["k_num"] == kv)].copy()
                if d_km.empty:
                    continue

                y = d_km["ConvergenceRate"].astype(float).to_numpy()
                jitter = RNG.normal(loc=0.0, scale=0.06, size=y.size)
                x_center = base[i_k] + offset
                xs = x_center + jitter

                is_xbar = d_km["point_id"].astype(str).str.lower() == "xbar"

                if is_xbar.any():
                    ax.scatter(
                        xs[is_xbar.to_numpy()],
                        y[is_xbar.to_numpy()],
                        s=110,
                        marker="X",
                        color=c,
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=3,
                    )

                if (~is_xbar).any():
                    ax.scatter(
                        xs[(~is_xbar).to_numpy()],
                        y[(~is_xbar).to_numpy()],
                        s=75,
                        marker="o",
                        color=c,
                        edgecolors="black",
                        linewidths=0.9,
                        zorder=3,
                    )

        add_reference_lines(ax)
        set_axes_style(ax)

        hm_clean = str(hm).strip()
        ax.set_title(f"h = {hm_clean}" if hm_clean else "h", fontsize=SUBTITLE_FONTSIZE, pad=10)

        ax.set_xlabel("k", fontsize=LABEL_FONTSIZE)
        ax.set_xticks(base, [str(int(k)) if float(k).is_integer() else str(k) for k in k_vals])

        ax.set_ylim(bottom=y0, top=y1)

    axes[0].set_ylabel("convergence rate q", fontsize=LABEL_FONTSIZE)

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=1.02)

    handles = make_legend_handles()
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        framealpha=0.95,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    ensure_dir(outpath_png.parent)
    fig.savefig(outpath_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# executes plots
def main(csv_path=DEFAULT_CSV_PATH, out_root=DEFAULT_OUT_ROOT):
    if not csv_path.exists():
        print(f"[rate_of_converges.py] csv not found: {csv_path}")
        return 1

    # read all data
    df = pd.read_csv(csv_path)

    # check valid structure of csv file
    required = ["Problem", "n", "Method", "Case", "Success", "ConvergenceRate", "point_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[rate_of_converges.py] missing columns in csv: {missing}")
        return 2

    df = df.copy()

    #check string format
    for c in ["Problem", "Method", "Case", "h_mode", "start_type", "point_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # only converged sequences are kept
    df["Success"] = df["Success"].apply(to_bool_success)
    df = df[df["Success"]]

    # check numeric format
    df["ConvergenceRate"] = numeric_series(df["ConvergenceRate"])
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df[np.isfinite(df["n"].to_numpy())]
    df["n"] = df["n"].astype(int)

    # drop non-finite q
    df = df[np.isfinite(df["ConvergenceRate"].to_numpy())]

    # drop outliers
    before = len(df)
    df = df[(df["ConvergenceRate"] >= MIN_Q) & (df["ConvergenceRate"] <= MAX_Q)]
    removed = before - len(df)
    if removed > 0:
        print(f"[rate_of_converges.py] dropped {removed} outlier q values outside [{MIN_Q}, {MAX_Q}]")

    if df.empty:
        print("[rate_of_converges.py] no successful runs with finite ConvergenceRate after filtering")
        return 0

    df["Problem"] = df["Problem"].astype(str)
    df["Case"] = df["Case"].apply(pretty_case)

    # group by problems, dimensio and case 
    problems = sorted(df["Problem"].unique().tolist())
    for prob in problems:
        d_p = df[df["Problem"] == prob]
        ns = sorted(d_p["n"].unique().tolist())
        for n in ns:
            d_n = d_p[d_p["n"] == n]
            cases = sorted(d_n["Case"].unique().tolist())
            for case in cases:
                d_c = d_n[d_n["Case"] == case].copy()
                if d_c.empty:
                    continue

                prob_dir = out_root / filename(prob)
                ensure_dir(prob_dir)

                case_slug = filename(case)
                out_png = prob_dir / f"n_{n}_{case_slug}.png"

                title = f"{prob} | n={n} | {case} | Experimental convergence rate q"

                if case.strip().lower() == "exact":
                    plot_exact_case(d_c, title=title, outpath_png=out_png)
                else:
                    plot_fd_case(d_c, title=title, outpath_png=out_png)

                print(f"[rate_of_converges.py] saved: {out_png}")

    return 0


if __name__ == "__main__":
    main()