"""Results & Discussion -- paper-ready figures and tables.

MSc Research Paper: Multi-Agent LLM Expert Systems for Bayesian Experimental Design
Ramon Lopez Nieto, Imperial College London, 2026

This is the canonical figure/table generation script: every figure and table
file it writes is the one referenced by ``\\includegraphics`` in the paper's
LaTeX source (see ``section4_results.tex``). It consumes only the pre-computed
checkpoint/result files in ``experiments/`` (the outputs of the notebooks in
that same directory) -- it does not call any LLM or rerun any Bayesian
optimisation campaign, so it runs in seconds on a laptop.

This is a behaviour-preserving refactor of the original ``results_plots.py``:
the only change from the original is how input/output directories are
resolved (hardcoded absolute Windows paths -> paths relative to the repo
root, overridable via environment variables). Every computation, statistical
test, plot call, and output filename is unchanged.

Usage
-----
    python scripts/generate_figures.py

Environment variables (optional)
---------------------------------
    BAYESLLM_DATA_DIR   Directory containing the checkpoint/.csv inputs
                         (default: ``<repo_root>/experiments``)
    BAYESLLM_FIGURES_DIR  Output directory for figures/tables
                         (default: ``<repo_root>/figures``)
"""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, fisher_exact

# -- Paths --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
UP = Path(os.environ.get("BAYESLLM_DATA_DIR", REPO_ROOT / "experiments"))
OUT = Path(os.environ.get("BAYESLLM_FIGURES_DIR", REPO_ROOT / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

PALETTE = {
    "GP-BO": "#2C3E50", "SingleAgent": "#8E44AD", "PCV": "#E74C3C",
    "BatchSelect": "#3498DB", "Batch-Random": "#95A5A6", "Batch-TopAcq": "#27AE60",
    "KernelPrior": "#E67E22", "KernelPrior-Loop": "#F39C12",
    "BatchSelect-RAG": "#1ABC9C", "SafeAgent": "#27AE60", "SafePCV": "#E74C3C",
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


def load_pkl(name: str) -> pd.DataFrame:
    """Load a pickled list/records checkpoint from ``UP`` into a DataFrame."""
    with open(UP / name, "rb") as f:
        return pd.DataFrame(pickle.load(f))


def best_so_far(df: pd.DataFrame, method_col: str, yield_col: str = "yield") -> pd.DataFrame:
    """Add a cumulative-max 'best_so_far' column per (method, seed) trajectory."""
    iter_cols = [c for c in df.columns if "iter" in c.lower()]
    iter_col = iter_cols[0] if iter_cols else "iteration"
    df = df.sort_values([method_col, "seed", iter_col])
    df["best_so_far"] = df.groupby([method_col, "seed"])[yield_col].cummax()
    return df


def convergence_stats(df, method_col, iter_col, methods_order=None):
    """Per-iteration mean and 95% CI of best-so-far, for each method."""
    if methods_order is None:
        methods_order = sorted(df[method_col].unique())
    stats = {}
    for m in methods_order:
        sub = df[df[method_col] == m]
        iters = sorted(sub[iter_col].unique())
        means, los, his = [], [], []
        for it in iters:
            vals = sub[sub[iter_col] == it]["best_so_far"].values
            mu = np.mean(vals)
            ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
            means.append(mu); los.append(mu - ci); his.append(mu + ci)
        stats[m] = {"iters": iters, "mean": means, "lo": los, "hi": his}
    return stats


def plot_convergence(stats, rename, fname, ylabel="Best-so-far yield (%)", palette=PALETTE):
    """Line + CI-band convergence plot, one line per method, saved to OUT/fname."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, s in stats.items():
        label = rename.get(method, method)
        color = palette.get(label, "#333333")
        ax.plot(s["iters"], s["mean"], label=label, color=color, linewidth=1.8)
        ax.fill_between(s["iters"], s["lo"], s["hi"], alpha=0.15, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / fname)
    plt.close(fig)
    print(f"  Saved {fname}")


def final_yields(df, method_col, iter_col):
    """Last-iteration best_so_far row per (method, seed)."""
    idx = df.groupby([method_col, "seed"])[iter_col].idxmax()
    return df.loc[idx, [method_col, "seed", "best_so_far"]].copy()


def wilcoxon_table(finals, method_col, methods=None):
    """Pairwise paired two-sided Wilcoxon signed-rank test across all method pairs."""
    if methods is None:
        methods = sorted(finals[method_col].unique())
    rows = []
    for m1, m2 in combinations(methods, 2):
        v1 = finals[finals[method_col] == m1].sort_values("seed")["best_so_far"].values
        v2 = finals[finals[method_col] == m2].sort_values("seed")["best_so_far"].values
        n = min(len(v1), len(v2))
        v1, v2 = v1[:n], v2[:n]
        try:
            stat, p = wilcoxon(v1, v2)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({"Method 1": m1, "Method 2": m2,
                      "Mean 1": np.mean(v1), "Mean 2": np.mean(v2),
                      "Delta": np.mean(v1) - np.mean(v2),
                      "W-stat": stat, "p-value": p,
                      "Significant": "Yes" if p < 0.05 else "No"})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: CRITIC FALLBACK RATES
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 1 --")
df_r1 = load_pkl("seeds_loop_checkpoint.pkl")
df_r2 = load_pkl("seeds_loop_checkpoint_run2.pkl")

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
for m in methods_fb:
    sub = df_r1[df_r1["method"] == m]
    print(f"{m}: fuente values = {sub['fuente'].unique() if 'fuente' in sub.columns else 'NO COLUMN'}")
    if "rechazos" in sub.columns:
        print(f"  rechazos==3: {(sub['rechazos'] == 3).mean() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(methods_fb))
w = 0.35
bars1 = ax.bar(x - w/2, [fb1[m] for m in methods_fb], w,
               label="Run 1 (with redundancy check)", color="#E74C3C", alpha=0.8)
bars2 = ax.bar(x + w/2, [fb2[m] for m in methods_fb], w,
               label="Run 2 (redundancy check removed)", color="#27AE60", alpha=0.8)
ax.set_ylabel("Fallback rate (%)")
ax.set_xticks(x)
ax.set_xticklabels([RENAME_BH.get(m, m) for m in methods_fb], rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
fig.savefig(OUT / "fig1_fallback_rates.png")
plt.close(fig)
print("  Saved fig1_fallback_rates.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: RUN 1 vs RUN 2 CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 2 --")
for df_run, fname in [(df_r1, "fig2a_convergence_run1.png"),
                       (df_r2, "fig2b_convergence_run2.png")]:
    df_run = best_so_far(df_run, "method", "yield")
    stats = convergence_stats(df_run, "method", "iteracion",
               ["bo_puro","single_agent","multiagente","batch_llm","batch_random","batch_top_acquisition"])
    plot_convergence(stats, RENAME_BH, fname)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: 20-SEED EI vs UCB CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 3 --")
df20 = pd.read_csv(UP / "all_results_full_grid_20seeds.csv")
ei_methods = ["bo_puro", "multiagente", "batch_llm"]
ucb_methods = ["bo_puro_ucb", "multiagente_ucb", "batch_llm_ucb"]

for methods, suffix in [(ei_methods, "ei"), (ucb_methods, "ucb")]:
    df_sub = df20[df20["method"].isin(methods)].copy()
    df_sub = best_so_far(df_sub, "method")
    stats = convergence_stats(df_sub, "method", "iteracion", methods)
    plot_convergence(stats, RENAME_BH, f"fig3_{suffix}_convergence_20seeds.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: FINAL YIELD BOXPLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 4 --")
df20 = best_so_far(df20, "method")
finals_20 = final_yields(df20, "method", "iteracion")
finals_20["Acquisition"] = finals_20["method"].apply(lambda m: "UCB" if "ucb" in m else "EI")
finals_20["method_clean"] = finals_20["method"].map(RENAME_BH)

fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["GP-BO", "PCV", "BatchSelect"]
sns.boxplot(data=finals_20, x="method_clean", y="best_so_far", hue="Acquisition",
            order=order, palette={"EI": "#3498DB", "UCB": "#E67E22"}, width=0.6, ax=ax)
sns.stripplot(data=finals_20, x="method_clean", y="best_so_far", hue="Acquisition",
              order=order, palette={"EI": "#3498DB", "UCB": "#E67E22"},
              dodge=True, alpha=0.4, size=4, ax=ax, legend=False)
ax.set_xlabel("Method")
ax.set_ylabel("Final best-so-far yield (%)")
ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig4_boxplot_ei_ucb_20seeds.png")
plt.close(fig)
print("  Saved fig4_boxplot_ei_ucb_20seeds.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: PAIRED DIFFERENCE PLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 5 --")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for idx, (acq_label, methods, ref) in enumerate([
    ("EI", ["multiagente", "batch_llm"], "bo_puro"),
    ("UCB", ["multiagente_ucb", "batch_llm_ucb"], "bo_puro_ucb")
]):
    ax = axes[idx]
    ref_vals = finals_20[finals_20["method"] == ref].sort_values("seed")["best_so_far"].values
    for m in methods:
        m_vals = finals_20[finals_20["method"] == m].sort_values("seed")["best_so_far"].values
        n = min(len(ref_vals), len(m_vals))
        diffs = m_vals[:n] - ref_vals[:n]
        label = RENAME_BH.get(m, m)
        color = PALETTE.get(label, "#333")
        ax.scatter(range(n), diffs, label=label, color=color, s=40, alpha=0.7, zorder=3)
        ax.plot(range(n), diffs, color=color, alpha=0.3, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Seed")
    if idx == 0:
        ax.set_ylabel("Yield difference (method minus GP-BO)")
    ax.text(0.02, 0.98, acq_label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
fig.savefig(OUT / "fig5_paired_difference.png")
plt.close(fig)
print("  Saved fig5_paired_difference.png")


# ══════════════════════════════════════════════════════════════════════════
# TABLE 1: WILCOXON (20 seeds)
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Table 1 --")
for acq, methods in [("EI", ei_methods), ("UCB", ucb_methods)]:
    wt = wilcoxon_table(finals_20, "method", methods)
    wt["Method 1"] = wt["Method 1"].map(RENAME_BH)
    wt["Method 2"] = wt["Method 2"].map(RENAME_BH)
    print(f"\n  {acq}:")
    print(wt.to_string(index=False))
    wt.to_csv(OUT / f"table1_wilcoxon_{acq.lower()}_20seeds.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: DIRECT ARYLATION (n=20)
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 6 --")
df_ar = load_pkl("results_n20_arylation.pkl")
df_ar = best_so_far(df_ar, "method")

for methods, suffix in [
    (["bo_only","multiagent","batch_llm"], "ei"),
    (["bo_only_ucb","multiagent_ucb","batch_llm_ucb"], "ucb")
]:
    df_sub = df_ar[df_ar["method"].isin(methods)].copy()
    stats = convergence_stats(df_sub, "method", "iteration", methods)
    plot_convergence(stats, RENAME_AR, f"fig6_arylation_{suffix}_20seeds.png")

finals_ar = final_yields(df_ar, "method", "iteration")
for acq, methods in [("EI", ["bo_only","multiagent","batch_llm"]),
                      ("UCB", ["bo_only_ucb","multiagent_ucb","batch_llm_ucb"])]:
    wt = wilcoxon_table(finals_ar, "method", methods)
    wt["Method 1"] = wt["Method 1"].map(RENAME_AR)
    wt["Method 2"] = wt["Method 2"].map(RENAME_AR)
    print(f"\n  Arylation {acq}:")
    print(wt.to_string(index=False))
    wt.to_csv(OUT / f"table_wilcoxon_arylation_{acq.lower()}.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: KERNEL SHAPING (n=12)
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 7 --")
df_ks = load_pkl("checkpoint_comparacion_kernel_shaping.pkl")
df_ks = df_ks.rename(columns={"metodo": "method"})
if "best_so_far" not in df_ks.columns:
    df_ks = best_so_far(df_ks, "method")

ks_methods = ["bo_puro", "bo_kernel_shaped", "bo_kernel_shaped_ard_loop", "batch_llm_grounded"]
stats = convergence_stats(df_ks, "method", "iteracion", ks_methods)
plot_convergence(stats, RENAME_KS, "fig7_kernel_shaping.png")

finals_ks = final_yields(df_ks, "method", "iteracion")
wt_ks = wilcoxon_table(finals_ks, "method", ks_methods)
wt_ks["Method 1"] = wt_ks["Method 1"].map(RENAME_KS)
wt_ks["Method 2"] = wt_ks["Method 2"].map(RENAME_KS)
print("\n  Kernel shaping Wilcoxon:")
print(wt_ks.to_string(index=False))
wt_ks.to_csv(OUT / "table_wilcoxon_kernel_shaping.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: MODEL SCALE
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 8 --")
df_7b = best_so_far(df_r2.copy(), "method")
df_30b = best_so_far(load_pkl("seeds_loop_checkpoint_qwen30b.pkl"), "method")

finals_7b = final_yields(df_7b, "method", "iteracion")
finals_30b = final_yields(df_30b, "method", "iteracion")
finals_7b["Model"] = "qwen2.5:7b"
finals_30b["Model"] = "qwen3:30b"
both = pd.concat([finals_7b, finals_30b])
both["method_clean"] = both["method"].map(RENAME_BH)

fig, ax = plt.subplots(figsize=(10, 4.5))
methods_order = ["GP-BO", "SingleAgent", "PCV", "BatchSelect", "Batch-Random", "Batch-TopAcq"]
sns.boxplot(data=both, x="method_clean", y="best_so_far", hue="Model",
            order=methods_order, palette={"qwen2.5:7b": "#3498DB", "qwen3:30b": "#E74C3C"},
            width=0.6, ax=ax)
ax.set_xlabel("Method")
ax.set_ylabel("Final best-so-far yield (%)")
ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig8_model_scale.png")
plt.close(fig)
print("  Saved fig8_model_scale.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9: SAFETY VIOLATION RATES
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 9 --")
df_safety = load_pkl("checkpoint_safety_multiagent_snar.pkl")
safety_methods = ["bo_puro", "single_agent", "single_agent_safety_grounded", "multiagent_specialists"]

viol_data = []
for m in safety_methods:
    sub = df_safety[df_safety["method"] == m]
    viol_data.append({"method": RENAME_SAFETY[m],
                       "violation_rate": sub["is_unsafe"].mean() * 100,
                       "n_unsafe": int(sub["is_unsafe"].sum()),
                       "n_total": len(sub)})
viol_df = pd.DataFrame(viol_data)

fig, ax = plt.subplots(figsize=(6, 4))
colors = [PALETTE.get(m, "#333") for m in viol_df["method"]]
bars = ax.bar(viol_df["method"], viol_df["violation_rate"], color=colors, alpha=0.85, edgecolor="white")
for bar, row in zip(bars, viol_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{row['n_unsafe']}/{row['n_total']}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Safety violation rate (%)")
ax.set_xlabel("Method")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 35)
fig.savefig(OUT / "fig9_safety_violations.png")
plt.close(fig)
print("  Saved fig9_safety_violations.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10: SAFETY PARETO
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 10 --")
pareto_data = []
for m in safety_methods:
    sub = df_safety[df_safety["method"] == m]
    pareto_data.append({"method": RENAME_SAFETY[m],
                         "mean_sty": sub["sty"].mean(),
                         "violation_rate": sub["is_unsafe"].mean() * 100})
pareto_df = pd.DataFrame(pareto_data)

fig, ax = plt.subplots(figsize=(6, 4.5))
for _, row in pareto_df.iterrows():
    color = PALETTE.get(row["method"], "#333")
    ax.scatter(row["mean_sty"], row["violation_rate"], s=150, c=color,
               edgecolors="black", linewidth=0.8, zorder=5)
    ax.annotate(row["method"], (row["mean_sty"], row["violation_rate"]),
                textcoords="offset points", xytext=(15, 5), fontsize=9,
                fontweight="bold", color=color)
ax.set_xlabel("Mean space-time yield (STY)")
ax.set_ylabel("Safety violation rate (%)")
ax.grid(True, alpha=0.3)
ax.annotate("", xy=(8500, 0), xytext=(500, 28),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.5, ls="--"))
ax.text(4500, 15, "Ideal\ndirection", fontsize=8, color="grey", ha="center", style="italic")
fig.savefig(OUT / "fig10_safety_pareto.png")
plt.close(fig)
print("  Saved fig10_safety_pareto.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 11: SAFETY STY CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 11 --")
df_safety_bsf = df_safety.copy().sort_values(["method", "seed", "iteration"])
df_safety_bsf["best_so_far"] = df_safety_bsf.groupby(["method", "seed"])["sty"].cummax()
stats_safety = convergence_stats(df_safety_bsf, "method", "iteration", safety_methods)
plot_convergence(stats_safety, RENAME_SAFETY, "fig11_safety_sty_convergence.png",
                 ylabel=r"Best-so-far STY (kg m$^{-3}$ h$^{-1}$)")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 12: DESCRIPTOR EMPHASIS MISMATCH
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 12 --")
desc_df = pd.DataFrame({
    "Descriptor": ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
                    "NumRotBonds", "MaxPartialChg", "MinPartialChg"],
    "BatchSelect (%)": [50.7, 43.0, 49.3, 2.0, 1.7, 25.7, 6.0, 0.3],
    "PCV (%)": [34.7, 56.7, 28.3, 0.0, 0.3, 14.7, 8.7, 0.7],
})

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(desc_df))
w = 0.3
ax.bar(x - w/2, desc_df["BatchSelect (%)"], w, label="BatchSelect", color="#3498DB", alpha=0.85)
ax.bar(x + w/2, desc_df["PCV (%)"], w, label="PCV", color="#E74C3C", alpha=0.85)
for i in [6, 7]:
    ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color="#27AE60")
ax.set_xticks(x)
ax.set_xticklabels(desc_df["Descriptor"], rotation=30, ha="right")
ax.set_ylabel("Mention rate in reasoning texts (%)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.annotate("Most predictive\n(largest $R^2$ gain)", xy=(6.5, 10), fontsize=8,
            color="#27AE60", fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#27AE60", alpha=0.1))
fig.savefig(OUT / "fig12_descriptor_mismatch.png")
plt.close(fig)
print("  Saved fig12_descriptor_mismatch.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 13: EI vs UCB HEATMAP
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 13 --")
heatmap_data = []
for m in ei_methods + ucb_methods:
    sub = finals_20[finals_20["method"] == m]
    acq = "UCB" if "ucb" in m else "EI"
    heatmap_data.append({"Method": RENAME_BH[m], "Acquisition": acq,
                          "Mean Yield": sub["best_so_far"].mean()})
hm_pivot = pd.DataFrame(heatmap_data).pivot(index="Method", columns="Acquisition", values="Mean Yield")
hm_pivot = hm_pivot.loc[["GP-BO", "PCV", "BatchSelect"], ["EI", "UCB"]]

fig, ax = plt.subplots(figsize=(4.5, 3))
sns.heatmap(hm_pivot, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=1,
            linecolor="white", ax=ax, vmin=82, vmax=87,
            cbar_kws={"label": "Mean final yield (%)"})
fig.savefig(OUT / "fig13_ei_ucb_heatmap.png")
plt.close(fig)
print("  Saved fig13_ei_ucb_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 14: CROSS-BENCHMARK
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 14 --")
cross_data = []
for m_bh, m_ar in [("bo_puro","bo_only"), ("multiagente","multiagent"), ("batch_llm","batch_llm")]:
    label = RENAME_BH[m_bh]
    for v in finals_20[finals_20["method"]==m_bh]["best_so_far"].values:
        cross_data.append({"Method": label, "Benchmark": "Buchwald\nHartwig", "Yield": v})
    for v in finals_ar[finals_ar["method"]==m_ar]["best_so_far"].values:
        cross_data.append({"Method": label, "Benchmark": "Direct\nArylation", "Yield": v})

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=pd.DataFrame(cross_data), x="Method", y="Yield", hue="Benchmark",
            palette={"Buchwald\nHartwig": "#3498DB", "Direct\nArylation": "#E67E22"},
            width=0.6, ax=ax)
ax.set_ylabel("Final best-so-far yield (%)")
ax.set_xlabel("Method")
ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig14_cross_benchmark.png")
plt.close(fig)
print("  Saved fig14_cross_benchmark.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 15: PER-SEED SAFETY VIOLATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 15 --")
seed_viol = df_safety.groupby(["method", "seed"])["is_unsafe"].sum().reset_index()
seed_viol["method_clean"] = seed_viol["method"].map(RENAME_SAFETY)

fig, ax = plt.subplots(figsize=(7, 4))
order = ["GP-BO", "SingleAgent", "SafeAgent", "SafePCV"]
sns.stripplot(data=seed_viol, x="method_clean", y="is_unsafe", hue="method_clean",
              order=order, palette=PALETTE, size=10, alpha=0.7, jitter=0.15, ax=ax, legend=False)
ax.set_ylabel("Unsafe iterations per seed (out of 15)")
ax.set_xlabel("Method")
ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig15_perseed_violations.png")
plt.close(fig)
print("  Saved fig15_perseed_violations.png")


# ══════════════════════════════════════════════════════════════════════════
# TABLE 2: FISHER EXACT (SAFETY)
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Table 2 --")
fisher_rows = []
for m1, m2 in combinations(safety_methods, 2):
    s1 = df_safety[df_safety["method"] == m1]["is_unsafe"]
    s2 = df_safety[df_safety["method"] == m2]["is_unsafe"]
    table = [[s1.sum(), len(s1) - s1.sum()], [s2.sum(), len(s2) - s2.sum()]]
    odds, p = fisher_exact(table)
    fisher_rows.append({"Method 1": RENAME_SAFETY[m1], "Method 2": RENAME_SAFETY[m2],
                         "Violations 1": f"{int(s1.sum())}/{len(s1)}",
                         "Violations 2": f"{int(s2.sum())}/{len(s2)}",
                         "Odds ratio": f"{odds:.2f}", "p-value": f"{p:.6f}",
                         "Significant": "Yes" if p < 0.05 else "No"})
fisher_df = pd.DataFrame(fisher_rows)
print(fisher_df.to_string(index=False))
fisher_df.to_csv(OUT / "table2_fisher_safety.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
# TABLE 3: COMPREHENSIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Table 3 --")
summary_rows = []
for m in ei_methods:
    vals = finals_20[finals_20["method"]==m]["best_so_far"]
    summary_rows.append({"Benchmark":"Buchwald Hartwig","Acq":"EI",
                          "Method":RENAME_BH[m],"n":20,
                          "Mean":f"{vals.mean():.2f}","Std":f"{vals.std():.2f}"})
for m in ucb_methods:
    vals = finals_20[finals_20["method"]==m]["best_so_far"]
    summary_rows.append({"Benchmark":"Buchwald Hartwig","Acq":"UCB",
                          "Method":RENAME_BH[m],"n":20,
                          "Mean":f"{vals.mean():.2f}","Std":f"{vals.std():.2f}"})
for m in ["bo_only","multiagent","batch_llm"]:
    vals = finals_ar[finals_ar["method"]==m]["best_so_far"]
    summary_rows.append({"Benchmark":"Direct Arylation","Acq":"EI",
                          "Method":RENAME_AR[m],"n":20,
                          "Mean":f"{vals.mean():.2f}","Std":f"{vals.std():.2f}"})
for m in safety_methods:
    sub = df_safety[df_safety["method"]==m]
    summary_rows.append({"Benchmark":"SNAr (safety)","Acq":"EI",
                          "Method":RENAME_SAFETY[m],"n":7,
                          "Mean":f"{sub['sty'].mean():.0f}",
                          "Std":f"Viol: {sub['is_unsafe'].mean()*100:.1f}%"})
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT / "table3_comprehensive_summary.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 16: MODEL THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Figure 16 --")
mb = pd.read_csv(UP / "model_benchmark_summary.csv")
mb["label"] = mb["model"] + " (" + mb["rol"] + ")"

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
mb1 = mb.sort_values("tokens_s_media", ascending=True)
c1 = ["#3498DB" if "qwen2.5" in m else "#E74C3C" for m in mb1["model"]]
axes[0].barh(mb1["label"], mb1["tokens_s_media"], color=c1, alpha=0.85,
             xerr=mb1["tokens_s_std"], capsize=3)
axes[0].set_xlabel("Tokens per second")
axes[0].grid(axis="x", alpha=0.3)

mb2 = mb.sort_values("latencia_media_s", ascending=False)
c2 = ["#3498DB" if "qwen2.5" in m else "#E74C3C" for m in mb2["model"]]
axes[1].barh(mb2["label"], mb2["latencia_media_s"], color=c2, alpha=0.85,
             xerr=mb2["latencia_std_s"], capsize=3)
axes[1].set_xlabel("Latency (seconds)")
axes[1].grid(axis="x", alpha=0.3)

fig.savefig(OUT / "fig16_model_throughput.png")
plt.close(fig)
print("  Saved fig16_model_throughput.png")


print(f"\n{'='*60}")
print(f"All figures and tables saved to {OUT}")
