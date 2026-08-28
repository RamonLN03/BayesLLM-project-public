"""Merged EI + UCB convergence figure (Buchwald-Hartwig, 20 seeds).

Panel (a): Expected Improvement. Panel (b): Upper Confidence Bound.

Uses the same data and method names as ``generate_figures_supplementary.py``:
``all_results_full_grid_20seeds.csv``. This two-panel vector figure is the
likely source of the polished "Figure 3 / yield hero" panel used in the main
body of the paper (``Figure3_yield_hero.pdf``); that exact PDF was hand
finished outside this repository, so this script is documented as the
best-available source rather than an exact byte-for-byte reproduction.

Behaviour-preserving refactor of the original ``replot_fig3,4.py``: only the
input/output path resolution changed. Every computation and plot call is
unchanged.

Usage
-----
    python scripts/generate_figure3_hero.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- Paths --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BAYESLLM_DATA_DIR", REPO_ROOT / "experiments"))
DATA_FILE = DATA_DIR / "all_results_full_grid_20seeds.csv"

OUT_DIR = Path(os.environ.get("BAYESLLM_SUPPLEMENTARY_FIGURES_DIR", REPO_ROOT / "figures_supplementary"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "fig3_bh_convergence.pdf"


# -- Figure dimensions --------------------------------------------------
# 180 mm = approximately full text width in a two-column paper
MM = 1 / 25.4
FIG_W = 180 * MM
FIG_H = 55 * MM


# -- Publication style ----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",

    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "figure.dpi": 300,
    "savefig.dpi": 300,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


# -- Colours / method names ------------------------------------------------
COLORS = {
    "GP-BO": "#2C3E50",
    "PCV": "#C0392B",
    "BatchSelect": "#2980B9",
}

ORDER = [
    "GP-BO",
    "PCV",
    "BatchSelect",
]

# Names used in the stored results
METHODS = {
    "EI": {
        "GP-BO": "bo_puro",
        "PCV": "multiagente",
        "BatchSelect": "batch_llm",
    },
    "UCB": {
        "GP-BO": "bo_puro_ucb",
        "PCV": "multiagente_ucb",
        "BatchSelect": "batch_llm_ucb",
    },
}


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

df = pd.read_csv(DATA_FILE)

required_columns = {"method", "seed", "iteracion", "yield"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns in {DATA_FILE}: {sorted(missing)}"
    )

# Sort before calculating cumulative best yield
df = df.sort_values(
    ["method", "seed", "iteracion"]
).copy()

# Convert raw yield trajectory into best-so-far trajectory
df["best_so_far"] = (
    df.groupby(["method", "seed"])["yield"]
      .cummax()
)


# =============================================================================
# HELPER: MEAN ± STANDARD ERROR AT EACH ITERATION
# =============================================================================

def convergence_stats(data, method_name):
    """
    Return iteration, mean best-so-far yield, and standard error
    for one optimisation method.
    """

    sub = data[data["method"] == method_name].copy()

    if sub.empty:
        raise ValueError(
            f"No data found for method '{method_name}'"
        )

    stats = (
        sub.groupby("iteracion")["best_so_far"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .sort_values("iteracion")
    )

    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    # If an iteration has only one observation, std/SE becomes NaN.
    stats["se"] = stats["se"].fillna(0.0)

    return (
        stats["iteracion"].to_numpy(),
        stats["mean"].to_numpy(),
        stats["se"].to_numpy(),
    )


# =============================================================================
# CREATE MERGED TWO-PANEL FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(FIG_W, FIG_H),
    sharey=True,
    constrained_layout=True,
)

for ax, acq, panel in zip(
    axes,
    ["EI", "UCB"],
    ["(a)", "(b)"],
):

    for clean_name in ORDER:

        raw_method = METHODS[acq][clean_name]

        iterations, mean_yield, se_yield = convergence_stats(
            df,
            raw_method,
        )

        color = COLORS[clean_name]

        # Mean best-so-far trajectory
        ax.plot(
            iterations,
            mean_yield,
            color=color,
            lw=1.4,
            label=clean_name,
            zorder=3,
        )

        # Mean ± standard error
        ax.fill_between(
            iterations,
            mean_yield - se_yield,
            mean_yield + se_yield,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    # Panel title
    ax.set_title(
        f"{panel} {acq}",
        fontsize=8.5,
        loc="left",
        pad=4,
    )

    ax.set_xlabel(
        "Iteration",
        fontsize=8,
    )

    ax.tick_params(
        axis="both",
        labelsize=7,
    )

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.5,
    )

    # Use the actual iteration range rather than assuming it starts at zero
    panel_methods = list(METHODS[acq].values())
    panel_data = df[df["method"].isin(panel_methods)]

    ax.set_xlim(
        panel_data["iteracion"].min(),
        panel_data["iteracion"].max(),
    )


# Shared y-axis label only on left panel
axes[0].set_ylabel(
    "Best-so-far yield (%)",
    fontsize=8,
)

# One legend is sufficient because methods are identical in both panels
axes[0].legend(
    loc="lower right",
    fontsize=7.5,
    frameon=False,
)


# =============================================================================
# SAVE
# =============================================================================

fig.savefig(
    OUT_FILE,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(fig)

print(f"Saved: {OUT_FILE}")
