# Validation report

This document records the check performed while preparing this public
repository: does the reorganised code reproduce the same numeric results and
figures as the original (private) working copy of the project?

## What was compared

The three figure/table-generation scripts (`results_plots.py`,
`results_plots_v2.py`, `replot_fig3,4.py` in the original project) were
rewritten as `scripts/generate_figures.py`, `scripts/generate_figures_supplementary.py`
and `scripts/generate_figure3_hero.py`. The only change made to these scripts
was how input/output directories are resolved: hardcoded absolute Windows
paths (e.g. `C:\Users\<user>\OneDrive\Documentos\bayesllm-project`) were
replaced with paths computed relative to the repository root (overridable via
environment variables). Every computation, statistical test, plotting call,
and output filename is unchanged from the original.

To check that this path change did not alter any result, the original
`figures/` and `figures2/` output directories (generated previously by the
original scripts, and copied into this repository unchanged as the
before-refactor reference) were backed up, then the three new scripts were
re-run from scratch against the copied `experiments/` data, and every output
file was compared against the backed-up original.

## Result

- **All 7 statistical result tables are byte-for-byte identical**
  (`table1_wilcoxon_ei_20seeds.csv`, `table1_wilcoxon_ucb_20seeds.csv`,
  `table_wilcoxon_arylation_ei.csv`, `table_wilcoxon_arylation_ucb.csv`,
  `table_wilcoxon_kernel_shaping.csv`, `table2_fisher_safety.csv`,
  `table3_comprehensive_summary.csv`). These are the files carrying every
  Wilcoxon/Fisher p-value, mean, and standard deviation reported in the
  paper.
- The regenerated numbers were also cross-checked by hand against the paper
  text and match exactly, e.g. Table 3 reports GP-BO 86.22% vs PCV 82.95%
  final yield on Buchwald-Hartwig/EI, matching "82.9% against 86.2% final
  yield" in the abstract and "82.95% versus 86.22%" in
  `section4_results.tex`; the safety table reports GP-BO 26.7% vs SafeAgent
  4.0% violation rate, matching "cuts hazardous-region violations from 26.7%
  to 4.0%" in the abstract.
- **34 of 38 figure files are byte-for-byte identical** (19/19 in `figures/`
  minus the 2 noted below, 13/15 in `figures_supplementary/` minus the 2
  noted below, plus the hero PDF).
- **4 figures differ by a small number of bytes**: `fig4_boxplot_ei_ucb_20seeds.png`,
  `fig15_perseed_violations.png`, `fig5_violin_ei_ucb.png`,
  `fig11_safety_composite.png`. In every case the difference is confined to
  the horizontal jitter of individual points in a `seaborn.stripplot` overlay
  — the underlying data (box/violin statistics, y-values, annotated counts)
  is identical. This is because the original code never seeds NumPy's global
  random state before calling `stripplot`, so its jitter pattern was already
  non-deterministic across runs of the *original* script, before any
  refactoring. This is a pre-existing cosmetic property of the reference
  implementation, not a change introduced by this repository, and it affects
  no reported number.

## How to reproduce this check yourself

```bash
python scripts/generate_figures.py
python scripts/generate_figures_supplementary.py
python scripts/generate_figure3_hero.py
```

Compare the resulting `figures/` and `figures_supplementary/` directories
against any previously generated copy; the `.csv` tables should match
exactly, and the `.png`/`.pdf` figures should match exactly except for the
stripplot jitter noted above.

## Second pass: extracting shared code into classes

A follow-up pass refactored the notebooks themselves — factoring the code
that was duplicated across `dia10.ipynb`, `dia11_prompt_sensitivity.ipynb`,
and `dia12_ucb_vs_ei.ipynb` (renamed `buchwald_hartwig_core_grid.ipynb`,
`kernel_prior_bh.ipynb`, `ucb_vs_ei_bh.ipynb`) into
`bayesllm.buchwald_hartwig.BuchwaldHartwigBenchmark`, and the analogous code
in `dia13_aylation_benchmark.ipynb` (renamed
`direct_arylation_benchmark.ipynb`) into
`bayesllm.direct_arylation.DirectArylationBenchmark` — plus removing
notebooks confirmed non-load-bearing or redundant. This could not be
verified the same way as the scripts above (it touches the notebooks that
call a live LLM, which cannot be re-run end-to-end offline, and LLM sampling
is not bit-reproducible even when it can be), so a different, two-part
verification was used instead:

1. **Before extracting anything**, every function that looked duplicated
   across the original notebooks was diffed programmatically (Python's
   `difflib`, not eyeballing) to find out which were truly byte-identical.
   Only confirmed-identical code was merged into the shared classes; every
   function that differed even slightly (e.g. `call_llm`'s default model,
   which is `qwen3:30b` in two notebooks but `qwen2.5:7b` in
   `kernel_prior_bh.ipynb` as saved — see `experiments/README.md`) was kept
   notebook-specific rather than risk silently unifying different behaviour.
2. **After extraction**, the new classes were checked against the original
   inline code by direct execution against the real dataset:
   - Deterministic parts (data loading, featurisation, RandomForest
     emulator training, GP-BO/UCB candidate and batch proposal, prompt-text
     generation) were confirmed to produce numerically identical output —
     same R², same predictions, `torch.allclose` on proposed candidates for
     matching seeds, identical prompt strings — for both benchmarks.
   - The LLM-dependent PCV deliberation loop (accept / reject-then-retry
     with Constant-Liar fantasy points / exhaust-retries-and-fall-back) was
     checked for identical control flow by injecting the same scripted,
     deterministic fake-LLM response queue into both the old procedural
     code and the new class method, across all three branches, and
     comparing every field of the result (final candidate, yield, source,
     rejection count, rejection log) — for both benchmarks.
   - Each rewritten notebook was then smoke-tested end-to-end (shrunk
     seed/iteration counts, the same scripted fake LLM) to confirm the full
     notebook — configuration, orchestration glue, and checkpoint I/O —
     runs without error and produces the expected record count and method
     coverage.

All of the above passed. This is a different, weaker standard of evidence
than the byte-for-byte script comparison above (it cannot rule out every
possible mistake the way re-running the exact original code and diffing the
output can), which is why it is reported separately and in more procedural
detail here, rather than folded into the "Result" section above.

## Known reproducibility limitations (notebooks, not scripts)

The scripts above are fully deterministic and were exactly reproduced (see
above). Full reproducibility of the *notebooks* that originally produced the
checkpoint/result files they consume is a separate question, and is honestly
weaker in a few specific, documented cases — see
["Data lineage and reproducibility notes"](experiments/README.md#data-lineage-and-reproducibility-notes)
in `experiments/README.md`. In short: some notebooks were edited after the
run that produced their currently-shipped output file, and two of the eight
required result files (`seeds_loop_checkpoint.pkl`,
`seeds_loop_checkpoint_run2.pkl`) have no notebook in this repository that
writes them, most likely because the notebook that originally produced them
predates the earliest notebook still on disk. This does **not** affect the
validation above, since that check used the shipped result files directly
(the same files the paper's figures were generated from), not a re-run of
the notebooks.
