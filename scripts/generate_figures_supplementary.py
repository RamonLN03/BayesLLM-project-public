"""Results & Discussion -- upgraded, paper-ready figures (supplementary set).

MSc Research Paper: Multi-Agent LLM Expert Systems for Bayesian Experimental Design
Ramon Lopez Nieto, Imperial College London, 2026

Generates a second, more heavily styled pass over the same result files as
``generate_figures.py`` (vector PDF/high-DPI PNG output, combined panels,
composite figures). These are the source figures behind the polished
"hero" panels used in the main body of the paper; the numbered
``fig1``..``fig16`` files in ``figures/`` (produced by ``generate_figures.py``)
are the ones directly cited by ``section4_results.tex``.

Behaviour-preserving refactor of the original ``results_plots_v2.py``: only
the input/output path resolution changed (hardcoded/`__file__`-relative
Windows paths -> paths relative to the repo root, overridable via
environment variables). Every computation and plot call is unchanged.

Usage
-----
    python scripts/generate_figures_supplementary.py
"""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import wilcoxon, fisher_exact

# -- Paths --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BAYESLLM_DATA_DIR", REPO_ROOT / "experiments"))
OUT = Path(os.environ.get("BAYESLLM_SUPPLEMENTARY_FIGURES_DIR", REPO_ROOT / "figures_supplementary"))
OUT.mkdir(parents=True, exist_ok=True)

# -- Style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
})

# Academic palette -- muted, distinguishable, colourblind-friendly
C = {
    "GP-BO":           "#2C3E50",
    "SingleAgent":     "#7D3C98",
    "PCV":             "#C0392B",
    "BatchSelect":     "#2980B9",
    "Batch-Random":    "#7F8C8D",
    "Batch-TopAcq":    "#27AE60",
    "KernelPrior":     "#D35400",
    "KernelPrior-Loop":"#F39C12",
    "BatchSelect-RAG": "#16A085",
    "SafeAgent":       "#27AE60",
    "SafePCV":         "#C0392B",
    "EI":              "#2980B9",
    "UCB":             "#D35400",
}

RENAME_BH = {
    "bo_puro": "GP-BO", "single_agent": "SingleAgent",
    "multiagente": "PCV", "batch_llm": "BatchSelect",
    "batch_random": "Batch-Random", "batch_top_acquisition": "Batch-TopAcq",
    "bo_puro_ucb": "GP-BO", "multiagente_ucb": "PCV", "batch_llm_ucb": "BatchSelect",
}
RENAME_AR = {
    "bo_only": "GP-BO", "multiagent": "PCV", "batch_llm": "BatchSelect",
    "bo_only_ucb": "GP-BO", "multiagent_ucb": "PCV", "batch_llm_ucb": "BatchSelect",
}
RENAME_SAFETY = {
    "bo_puro": "GP-BO", "single_agent": "SingleAgent",
    "single_agent_safety_grounded": "SafeAgent",
    "multiagent_specialists": "SafePCV",
}
RENAME_KS = {
    "bo_puro": "GP-BO", "bo_kernel_shaped": "KernelPrior",
    "bo_kernel_shaped_ard_loop": "KernelPrior-Loop",
    "batch_llm_grounded": "BatchSelect-RAG",
}


# -- Helpers ----------------------------------------------------------------
def load_pkl(name: str) -> pd.DataFrame:
    """Load a pickled list/records checkpoint from ``DATA_DIR`` into a DataFrame."""
    with open(DATA_DIR / name, "rb") as f:
        return pd.DataFrame(pickle.load(f))

def best_so_far(df: pd.DataFrame, method_col: str, yield_col: str = "yield") -> pd.DataFrame:
    """Add a cumulative-max 'best_so_far' column per (method, seed) trajectory."""
    iter_cols = [c for c in df.columns if "iter" in c.lower()]
    iter_col = iter_cols[0] if iter_cols else "iteration"
    df = df.sort_values([method_col, "seed", iter_col])
    df["best_so_far"] = df.groupby([method_col, "seed"])[yield_col].cummax()
    return df

def final_yields(df, method_col, iter_col):
    """Last-iteration best_so_far row per (method, seed)."""
    idx = df.groupby([method_col, "seed"])[iter_col].idxmax()
    return df.loc[idx, [method_col, "seed", "best_so_far"]].copy()

def convergence_stats(df, method_col, iter_col, methods_order):
    """Per-iteration mean and 95% CI (ddof=1) of best-so-far, for each method."""
    stats = {}
    for m in methods_order:
        sub = df[df[method_col] == m]
        iters = sorted(sub[iter_col].unique())
        means, los, his = [], [], []
        for it in iters:
            vals = sub[sub[iter_col] == it]["best_so_far"].values
            mu = np.mean(vals)
            ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))
            means.append(mu); los.append(mu - ci); his.append(mu + ci)
        stats[m] = {"iters": np.array(iters), "mean": np.array(means),
                     "lo": np.array(los), "hi": np.array(his)}
    return stats

def add_panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")

def savefig(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  Saved {name}")


# ══════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")
df_r1 = load_pkl("seeds_loop_checkpoint.pkl")
df_r2 = load_pkl("seeds_loop_checkpoint_run2.pkl")
df20 = pd.read_csv(DATA_DIR / "all_results_full_grid_20seeds.csv")
df_ar = load_pkl("results_n20_arylation.pkl")
df_ks = load_pkl("checkpoint_comparacion_kernel_shaping.pkl")
df_ks = df_ks.rename(columns={"metodo": "method"})
df_safety = load_pkl("checkpoint_safety_multiagent_snar.pkl")
mb = pd.read_csv(DATA_DIR / "model_benchmark_summary.csv")

# Pre-compute best-so-far
df_r2 = best_so_far(df_r2, "method", "yield")
df20 = best_so_far(df20, "method")
df_ar = best_so_far(df_ar, "method")
if "best_so_far" not in df_ks.columns:
    df_ks = best_so_far(df_ks, "method")

ei_methods = ["bo_puro", "multiagente", "batch_llm"]
ucb_methods = ["bo_puro_ucb", "multiagente_ucb", "batch_llm_ucb"]
safety_methods = ["bo_puro", "single_agent", "single_agent_safety_grounded", "multiagent_specialists"]
ks_methods = ["bo_puro", "bo_kernel_shaped", "bo_kernel_shaped_ard_loop", "batch_llm_grounded"]

finals_20 = final_yields(df20, "method", "iteracion")
finals_ar = final_yields(df_ar, "method", "iteration")
finals_ks = final_yields(df_ks, "method", "iteracion")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: CRITIC FALLBACK — HORIZONTAL LOLLIPOP
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 1: Critic fallback lollipop --")

def fallback_rate(df):
    rates = {}
    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m]
        if "rechazos" in sub.columns:
            fb = (sub["rechazos"] == 3).mean()
        elif "fuente" in sub.columns:
            fb = sub["fuente"].str.contains("fallback", case=False, na=False).mean()
        else:
            fb = 0.0
        rates[m] = fb * 100
    return rates

fb1 = fallback_rate(df_r1)
fb2 = fallback_rate(df_r2)
methods_fb = [m for m in fb1 if m != "bo_puro"]
labels = [RENAME_BH.get(m, m) for m in methods_fb]

fig, ax = plt.subplots(figsize=(7, 3.5))
y = np.arange(len(methods_fb))
for i, m in enumerate(methods_fb):
    # Run 1: full dot
    ax.plot(fb1[m], i, 'o', color="#C0392B", markersize=9, zorder=5)
    # Run 2: full dot
    ax.plot(fb2[m], i, 's', color="#27AE60", markersize=7, zorder=5)
    # connecting line
    ax.plot([fb2[m], fb1[m]], [i, i], color="#BDC3C7", linewidth=1.5, zorder=2)
    # annotation for Run 1
    ax.annotate(f"{fb1[m]:.0f}%", (fb1[m], i), textcoords="offset points",
                xytext=(8, 0), fontsize=7.5, color="#C0392B", va="center")

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Fallback rate (%)")
ax.set_xlim(-3, 105)
ax.legend([Line2D([0],[0],marker='o',color='#C0392B',ls='',ms=7),
           Line2D([0],[0],marker='s',color='#27AE60',ls='',ms=6)],
          ["Run 1 (with redundancy check)", "Run 2 (redundancy check removed)"],
          loc="lower right", framealpha=0.9, fontsize=7.5)
ax.grid(axis="x", alpha=0.2)
savefig(fig, "fig1_fallback_rates.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: RUN 2 CONVERGENCE (all 6 methods, 10 seeds)
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 2: Run 2 convergence --")
run2_methods = ["bo_puro","single_agent","multiagente","batch_llm","batch_random","batch_top_acquisition"]
stats_r2 = convergence_stats(df_r2, "method", "iteracion", run2_methods)

fig, ax = plt.subplots(figsize=(7, 4))
for m in run2_methods:
    s = stats_r2[m]
    label = RENAME_BH.get(m, m)
    color = C.get(label, "#333")
    ax.plot(s["iters"], s["mean"], label=label, color=color)
    ax.fill_between(s["iters"], s["lo"], s["hi"], alpha=0.1, color=color)
ax.set_xlabel("Iteration")
ax.set_ylabel("Best-so-far yield (%)")
ax.legend(loc="lower right", framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.15)
savefig(fig, "fig2_convergence_run2.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: EI vs UCB CONVERGENCE — SIDE BY SIDE PANELS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 3: EI vs UCB convergence (2 panels) --")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for idx, (methods, acq_label) in enumerate([(ei_methods, "EI"), (ucb_methods, "UCB")]):
    ax = axes[idx]
    df_sub = df20[df20["method"].isin(methods)].copy()
    stats = convergence_stats(df_sub, "method", "iteracion", methods)
    for m in methods:
        s = stats[m]
        label = RENAME_BH.get(m, m)
        color = C.get(label, "#333")
        ax.plot(s["iters"], s["mean"], label=label, color=color)
        ax.fill_between(s["iters"], s["lo"], s["hi"], alpha=0.12, color=color)
    ax.set_xlabel("Iteration")
    if idx == 0:
        ax.set_ylabel("Best-so-far yield (%)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.15)
    add_panel_label(ax, "AB"[idx])
    ax.set_title(f"{acq_label} ($n=20$ seeds)", fontsize=10)

savefig(fig, "fig3_ei_ucb_convergence.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: PAIRED DIFFERENCE — SLOPE/DUMBBELL PLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 4: Paired difference (dumbbell) --")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for idx, (acq_label, methods, ref) in enumerate([
    ("EI", ["multiagente", "batch_llm"], "bo_puro"),
    ("UCB", ["multiagente_ucb", "batch_llm_ucb"], "bo_puro_ucb"),
]):
    ax = axes[idx]
    ref_vals = finals_20[finals_20["method"] == ref].sort_values("seed")["best_so_far"].values
    seeds = np.arange(len(ref_vals))
    for m in methods:
        m_vals = finals_20[finals_20["method"] == m].sort_values("seed")["best_so_far"].values
        n = min(len(ref_vals), len(m_vals))
        diffs = m_vals[:n] - ref_vals[:n]
        label = RENAME_BH.get(m, m)
        color = C.get(label, "#333")
        # Stem lines
        for s_i in range(n):
            ax.plot([s_i, s_i], [0, diffs[s_i]], color=color, alpha=0.3, linewidth=0.8)
        ax.scatter(seeds[:n], diffs, label=label, color=color, s=35, alpha=0.8,
                   zorder=5, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Seed")
    if idx == 0:
        ax.set_ylabel("Yield difference (method $-$ GP-BO)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.15)
    add_panel_label(ax, "AB"[idx])
    ax.set_title(acq_label, fontsize=10, fontweight="bold")

savefig(fig, "fig4_paired_difference.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: VIOLIN + STRIP — EI vs UCB final yields
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 5: EI vs UCB violin --")
finals_20["Acquisition"] = finals_20["method"].apply(lambda m: "UCB" if "ucb" in m else "EI")
finals_20["method_clean"] = finals_20["method"].map(RENAME_BH)
plot_df = finals_20[finals_20["method_clean"].isin(["GP-BO","PCV","BatchSelect"])].copy()

fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["GP-BO", "PCV", "BatchSelect"]
sns.violinplot(data=plot_df, x="method_clean", y="best_so_far", hue="Acquisition",
               order=order, palette={"EI": C["EI"], "UCB": C["UCB"]},
               split=True, inner=None, alpha=0.3, cut=0, ax=ax)
sns.stripplot(data=plot_df, x="method_clean", y="best_so_far", hue="Acquisition",
              order=order, palette={"EI": C["EI"], "UCB": C["UCB"]},
              dodge=True, alpha=0.6, size=4, ax=ax, legend=False)
ax.set_xlabel("Method")
ax.set_ylabel("Final best-so-far yield (%)")
ax.grid(axis="y", alpha=0.15)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], framealpha=0.9)
savefig(fig, "fig5_violin_ei_ucb.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: EI vs UCB HEATMAP — annotated
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 6: EI/UCB heatmap --")
heatmap_data = []
for m in ei_methods + ucb_methods:
    sub = finals_20[finals_20["method"] == m]
    acq = "UCB" if "ucb" in m else "EI"
    heatmap_data.append({"Method": RENAME_BH[m], "Acquisition": acq,
                          "Mean Yield": sub["best_so_far"].mean()})
hm_pivot = pd.DataFrame(heatmap_data).pivot(index="Method", columns="Acquisition", values="Mean Yield")
hm_pivot = hm_pivot.loc[["GP-BO", "PCV", "BatchSelect"], ["EI", "UCB"]]

fig, ax = plt.subplots(figsize=(4, 2.8))
sns.heatmap(hm_pivot, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=1.5,
            linecolor="white", ax=ax, vmin=82, vmax=87,
            cbar_kws={"label": "Mean final yield (%)", "shrink": 0.8},
            annot_kws={"fontsize": 11, "fontweight": "bold"})
ax.set_ylabel("")
savefig(fig, "fig6_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: DIRECT ARYLATION — CONVERGENCE + CROSS-BENCHMARK COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 7: Direct Arylation --")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel A: convergence
ax = axes[0]
ar_ei_methods = ["bo_only", "multiagent", "batch_llm"]
df_ar_ei = df_ar[df_ar["method"].isin(ar_ei_methods)].copy()
stats = convergence_stats(df_ar_ei, "method", "iteration", ar_ei_methods)
for m in ar_ei_methods:
    s = stats[m]
    label = RENAME_AR.get(m, m)
    color = C.get(label, "#333")
    ax.plot(s["iters"], s["mean"], label=label, color=color)
    ax.fill_between(s["iters"], s["lo"], s["hi"], alpha=0.12, color=color)
ax.set_xlabel("Iteration")
ax.set_ylabel("Best-so-far yield (%)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_title("Direct Arylation, EI ($n=20$)", fontsize=10)
add_panel_label(ax, "A")

# Panel B: cross-benchmark comparison (dot plot with CI)
ax = axes[1]
cross_data = []
for bench_label, finals, rename, methods in [
    ("Buchwald–Hartwig", finals_20, RENAME_BH, ei_methods),
    ("Direct Arylation", finals_ar, RENAME_AR, ar_ei_methods),
]:
    for m in methods:
        sub = finals[finals["method"] == m]
        label = rename[m]
        cross_data.append({"Benchmark": bench_label, "Method": label,
                           "Mean": sub["best_so_far"].mean(),
                           "CI": 1.96*sub["best_so_far"].std()/np.sqrt(len(sub))})
cdf = pd.DataFrame(cross_data)
bench_names = cdf["Benchmark"].unique()
offsets = {"GP-BO": -0.15, "PCV": 0.0, "BatchSelect": 0.15}
for _, row in cdf.iterrows():
    x = list(bench_names).index(row["Benchmark"]) + offsets[row["Method"]]
    color = C[row["Method"]]
    ax.errorbar(x, row["Mean"], yerr=row["CI"], fmt='o', color=color,
                markersize=8, capsize=4, capthick=1.2, elinewidth=1.2,
                label=row["Method"] if row["Benchmark"]==bench_names[0] else "")
ax.set_xticks(range(len(bench_names)))
ax.set_xticklabels(bench_names)
ax.set_ylabel("Mean final yield (%)")
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.15)
ax.set_title("Cross-benchmark comparison, EI ($n=20$)", fontsize=10)
add_panel_label(ax, "B")

savefig(fig, "fig7_arylation_cross.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: KERNEL SHAPING — CONVERGENCE + WEIGHT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 8: Kernel shaping --")
fig, ax = plt.subplots(figsize=(7, 4))
stats_ks = convergence_stats(df_ks, "method", "iteracion", ks_methods)
for m in ks_methods:
    s = stats_ks[m]
    label = RENAME_KS.get(m, m)
    color = C.get(label, "#333")
    ax.plot(s["iters"], s["mean"], label=label, color=color)
    ax.fill_between(s["iters"], s["lo"], s["hi"], alpha=0.12, color=color)
ax.set_xlabel("Iteration")
ax.set_ylabel("Best-so-far yield (%)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.15)
savefig(fig, "fig8_kernel_shaping.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9: MODEL SCALE — PAIRED DOT PLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 9: Model scale --")
try:
    df_30b = best_so_far(load_pkl("seeds_loop_checkpoint_qwen30b.pkl"), "method")
    finals_7b = final_yields(df_r2, "method", "iteracion")
    finals_30b = final_yields(df_30b, "method", "iteracion")
    finals_7b["Model"] = "qwen2.5:7b"
    finals_30b["Model"] = "qwen3:30b"
    both = pd.concat([finals_7b, finals_30b])
    both["method_clean"] = both["method"].map(RENAME_BH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: boxplot comparison
    ax = axes[0]
    methods_order = ["GP-BO", "SingleAgent", "PCV", "BatchSelect", "Batch-Random", "Batch-TopAcq"]
    sns.boxplot(data=both, x="method_clean", y="best_so_far", hue="Model",
                order=methods_order, palette={"qwen2.5:7b": C["EI"], "qwen3:30b": C["PCV"]},
                width=0.6, ax=ax, fliersize=3)
    ax.set_xlabel("Method")
    ax.set_ylabel("Final best-so-far yield (%)")
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis="y", alpha=0.15)
    add_panel_label(ax, "A")

    # Panel B: throughput
    ax = axes[1]
    mb["label"] = mb["model"] + "\n(" + mb["rol"] + ")"
    mb1 = mb.sort_values("tokens_s_media", ascending=True)
    colors = [C["EI"] if "qwen2.5" in m else C["PCV"] for m in mb1["model"]]
    ax.barh(mb1["label"], mb1["tokens_s_media"], color=colors, alpha=0.85,
            xerr=mb1["tokens_s_std"], capsize=3, edgecolor="white")
    ax.set_xlabel("Tokens per second")
    ax.grid(axis="x", alpha=0.15)
    add_panel_label(ax, "B")

    fig.tight_layout()
    savefig(fig, "fig9_model_scale.png")
except FileNotFoundError:
    print("  SKIPPED: seeds_loop_checkpoint_qwen30b.pkl not found")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10: DESCRIPTOR EMPHASIS — DIVERGING BAR CHART
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 10: Descriptor emphasis --")
desc_df = pd.DataFrame({
    "Descriptor": ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
                    "NumRotBonds", "MaxPartialChg", "MinPartialChg"],
    "BatchSelect": [50.7, 43.0, 49.3, 2.0, 1.7, 25.7, 6.0, 0.3],
    "PCV":         [34.7, 56.7, 28.3, 0.0, 0.3, 14.7, 8.7, 0.7],
    "R2_gain":     [0, 0, 0, 0, 0, 0, 1, 1],  # 1 = most predictive
})
desc_df["avg_mention"] = (desc_df["BatchSelect"] + desc_df["PCV"]) / 2

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(desc_df))
w = 0.32

bars1 = ax.bar(x - w/2, desc_df["BatchSelect"], w, label="BatchSelect",
               color=C["BatchSelect"], alpha=0.85, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w/2, desc_df["PCV"], w, label="PCV",
               color=C["PCV"], alpha=0.85, edgecolor="white", linewidth=0.5)

# Highlight the most predictive descriptors
for i in [6, 7]:
    ax.axvspan(i - 0.48, i + 0.48, alpha=0.06, color="#27AE60", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(desc_df["Descriptor"], rotation=30, ha="right")
ax.set_ylabel("Mention rate in reasoning texts (%)")
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.15)

# Annotation
ax.annotate("Most predictive\n(largest $R^2$ gain)", xy=(6.5, 10), fontsize=8,
            color="#27AE60", fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#27AE60", alpha=0.08, edgecolor="none"))

savefig(fig, "fig10_descriptor_mismatch.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 11: SAFETY — 4-PANEL COMPOSITE
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 11: Safety composite --")
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

# Panel A: violation rate bar
ax = fig.add_subplot(gs[0, 0])
viol_data = []
for m in safety_methods:
    sub = df_safety[df_safety["method"] == m]
    viol_data.append({"method": RENAME_SAFETY[m],
                       "violation_rate": sub["is_unsafe"].mean() * 100,
                       "n_unsafe": int(sub["is_unsafe"].sum()),
                       "n_total": len(sub)})
viol_df = pd.DataFrame(viol_data)
colors_v = [C.get(m, "#333") for m in viol_df["method"]]
bars = ax.bar(viol_df["method"], viol_df["violation_rate"], color=colors_v,
              alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, row in zip(bars, viol_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{row['n_unsafe']}/{row['n_total']}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Safety violation rate (%)")
ax.set_ylim(0, 35)
ax.grid(axis="y", alpha=0.15)
add_panel_label(ax, "A")

# Panel B: Pareto trade-off
ax = fig.add_subplot(gs[0, 1])
pareto_data = []
for m in safety_methods:
    sub = df_safety[df_safety["method"] == m]
    pareto_data.append({"method": RENAME_SAFETY[m],
                         "mean_sty": sub["sty"].mean(),
                         "violation_rate": sub["is_unsafe"].mean() * 100})
for row in pareto_data:
    color = C.get(row["method"], "#333")
    ax.scatter(row["mean_sty"], row["violation_rate"], s=160, c=color,
               edgecolors="black", linewidth=0.6, zorder=5)
    offset = (15, 5) if row["method"] != "SafePCV" else (15, -10)
    ax.annotate(row["method"], (row["mean_sty"], row["violation_rate"]),
                textcoords="offset points", xytext=offset, fontsize=8.5,
                fontweight="bold", color=color)
ax.set_xlabel("Mean space-time yield (STY)")
ax.set_ylabel("Safety violation rate (%)")
ax.grid(True, alpha=0.15)
ax.annotate("", xy=(8500, 0), xytext=(500, 28),
            arrowprops=dict(arrowstyle="->", color="#BDC3C7", lw=1.2, ls="--"))
ax.text(4500, 15, "Ideal\ndirection", fontsize=8, color="#95A5A6", ha="center", style="italic")
add_panel_label(ax, "B")

# Panel C: STY convergence
ax = fig.add_subplot(gs[1, 0])
df_safety_bsf = df_safety.copy().sort_values(["method", "seed", "iteration"])
df_safety_bsf["best_so_far"] = df_safety_bsf.groupby(["method", "seed"])["sty"].cummax()
stats_s = convergence_stats(df_safety_bsf, "method", "iteration", safety_methods)
for m in safety_methods:
    s = stats_s[m]
    label = RENAME_SAFETY[m]
    color = C.get(label, "#333")
    ax.plot(s["iters"], s["mean"], label=label, color=color)
    ax.fill_between(s["iters"], s["lo"], s["hi"], alpha=0.12, color=color)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Best-so-far STY (kg m$^{-3}$ h$^{-1}$)")
ax.legend(loc="right", framealpha=0.9, fontsize=7)
ax.grid(True, alpha=0.15)
add_panel_label(ax, "C")

# Panel D: per-seed violations strip
ax = fig.add_subplot(gs[1, 1])
seed_viol = df_safety.groupby(["method", "seed"])["is_unsafe"].sum().reset_index()
seed_viol["method_clean"] = seed_viol["method"].map(RENAME_SAFETY)
order = ["GP-BO", "SingleAgent", "SafeAgent", "SafePCV"]
sns.stripplot(data=seed_viol, x="method_clean", y="is_unsafe", hue="method_clean",
              order=order, palette=C, size=8, alpha=0.65, jitter=0.12, ax=ax, legend=False)
# Add median line
for i, m in enumerate(order):
    vals = seed_viol[seed_viol["method_clean"]==m]["is_unsafe"]
    ax.plot([i-0.2, i+0.2], [vals.median(), vals.median()], color="black",
            linewidth=1.5, zorder=10)
ax.set_ylabel("Unsafe iterations per seed (out of 15)")
ax.set_xlabel("")
ax.grid(axis="y", alpha=0.15)
add_panel_label(ax, "D")

savefig(fig, "fig11_safety_composite.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 12: SAFETY CONDITIONS SCATTER — Temperature vs Concentration
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 12: Safety conditions scatter --")
# Extract conditions into columns
conds = pd.json_normalize(df_safety["final_conditions"])
df_sc = pd.concat([df_safety[["seed","method","iteration","sty","is_unsafe"]].reset_index(drop=True),
                    conds.reset_index(drop=True)], axis=1)
df_sc["method_clean"] = df_sc["method"].map(RENAME_SAFETY)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True, sharex=True)
for idx, m in enumerate(["GP-BO", "SingleAgent", "SafeAgent", "SafePCV"]):
    ax = axes[idx]
    sub = df_sc[df_sc["method_clean"] == m]
    safe = sub[~sub["is_unsafe"]]
    unsafe = sub[sub["is_unsafe"]]
    ax.scatter(safe["temperature"], safe["conc_dfnb"], s=18, alpha=0.5,
               color=C.get(m, "#333"), label="Safe", edgecolors="none")
    if len(unsafe) > 0:
        ax.scatter(unsafe["temperature"], unsafe["conc_dfnb"], s=30, alpha=0.8,
                   color="#E74C3C", marker="x", linewidths=1.2, label="Unsafe")
    ax.set_title(m, fontsize=9, fontweight="bold", color=C.get(m, "#333"))
    ax.set_xlabel("Temperature (°C)")
    if idx == 0:
        ax.set_ylabel("DFNB concentration (M)")
    ax.grid(True, alpha=0.15)
    if m == "GP-BO":
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

fig.tight_layout()
savefig(fig, "fig12_safety_conditions.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 13: COMPREHENSIVE SUMMARY HEATMAP
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 13: Summary table figure --")
# Build summary data
summary_vals = {}
for acq, methods, rename in [("EI", ei_methods, RENAME_BH), ("UCB", ucb_methods, RENAME_BH)]:
    for m in methods:
        label = rename[m]
        vals = finals_20[finals_20["method"]==m]["best_so_far"]
        summary_vals[(f"BH {acq}", label)] = vals.mean()

for m in ["bo_only", "multiagent", "batch_llm"]:
    label = RENAME_AR[m]
    vals = finals_ar[finals_ar["method"]==m]["best_so_far"]
    summary_vals[("DA EI", label)] = vals.mean()

benchmarks = ["BH EI", "BH UCB", "DA EI"]
methods_list = ["GP-BO", "PCV", "BatchSelect"]
matrix = np.zeros((len(methods_list), len(benchmarks)))
for i, m in enumerate(methods_list):
    for j, b in enumerate(benchmarks):
        matrix[i, j] = summary_vals.get((b, m), np.nan)

fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(pd.DataFrame(matrix, index=methods_list, columns=benchmarks),
            annot=True, fmt=".1f", cmap="RdYlGn", linewidths=1.5, linecolor="white",
            ax=ax, vmin=82, vmax=88,
            cbar_kws={"label": "Mean final yield (%)", "shrink": 0.8},
            annot_kws={"fontsize": 10, "fontweight": "bold"})
ax.set_ylabel("")
ax.set_title("Yield across benchmarks and acquisition functions", fontsize=10)
savefig(fig, "fig13_summary_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"All supplementary figures saved to {OUT}")
