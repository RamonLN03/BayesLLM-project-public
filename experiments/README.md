# Experiments

This directory holds the project's Jupyter notebooks together with the
checkpoint/result files they read and write, and the raw benchmark dataset.
They're kept together because the notebooks use working-directory-relative
file I/O (e.g. `pd.read_excel("data/...")`,
`pickle.load(open("seeds_loop_checkpoint.pkl"))`): splitting the code from
its data across directories would silently break those reads. When executed
via a job script, that working directory is set explicitly, see
`hpc/README.md`.

## Shared code

Six of the eight experiment notebooks build the same underlying pieces:
loading a chemistry dataset, training an emulator, fitting a GP and
optimising an acquisition function, running the Proposer-Critic-Verifier
(PCV) deliberation loop, with large stretches of literally duplicated code.
That shared logic now lives in the `bayesllm` package (`src/bayesllm/`,
installed with `pip install -e .`; see the root `README.md`):

- **`bayesllm.buchwald_hartwig.BuchwaldHartwigBenchmark`**: the Buchwald-Hartwig
  emulator plus every agent architecture run against it (GP-BO, SingleAgent,
  PCV, BatchSelect [+RAG-grounded variant], KernelPrior [+ARD-loop variant],
  and their UCB counterparts). Used by `buchwald_hartwig_core_grid.ipynb`,
  `kernel_prior_bh.ipynb`, and `ucb_vs_ei_bh.ipynb`.
- **`bayesllm.direct_arylation.DirectArylationBenchmark`**: the analogous
  class for the Direct Arylation benchmark (its own dataset, descriptor set,
  and prompts, but the same overall architecture). Used by
  `direct_arylation_benchmark.ipynb`.
- **`bayesllm.llm`**: the local-LLM calling convention (`/no_think` +
  `think=False` with a fallback, retried JSON parsing) shared by all of the
  above.

**How this was verified.** Before any extraction, I diffed every function
that looked duplicated across the original notebooks programmatically (not
by eye) to confirm it really was byte-identical; only confirmed-identical
code was merged. After extraction, I checked the class methods against the
original inline code by direct execution: the deterministic parts (data
loading, featurisation, emulator training, GP-BO candidate proposal) were
confirmed to produce numerically identical output for the same inputs/seeds,
and the LLM-dependent deliberation loop (PCV accept/reject/retry/fallback)
was checked for identical control flow using a scripted fake LLM response
queue fed into both the old and new code paths (the real LLM calls
themselves can't be re-run in an offline environment, and aren't
bit-reproducible even when they can). See `VALIDATION.md` for the full
before/after comparison this enabled.

**Two notebooks, `snar_rag_grounding.ipynb` and
`snar_safety_multiagent.ipynb`, deliberately do *not* use a shared class.**
Both work with the same SNAr kinetic simulator concept, but a diff I ran
before any refactoring decision showed their actual simulator/objective/agent
functions are *not* byte-identical (real, evolved differences, not
copy-paste duplication), see the per-notebook notes below. Forcing them
into one shared implementation would have risked silently changing one of
them, so each keeps its own direct, unmodified extraction of the original
notebook's logic instead.

## Notebook map

Renamed from the original numbered research-log notebooks ("dia" is Spanish
for "day") to names that describe what each one actually does. The mapping
from old to new name is given in each entry below, and the PBS job scripts
in `hpc/` were renamed and updated to match.

### Result-producing notebooks

| Notebook | Was | Produces | Feeds |
|---|---|---|---|
| `buchwald_hartwig_core_grid.ipynb` | `dia10.ipynb` | `seeds_loop_checkpoint_qwen30b.pkl` | Fig. 8 |
| `kernel_prior_bh.ipynb` | `dia11_prompt_sensitivity.ipynb` | `checkpoint_comparacion_kernel_shaping.pkl` | Fig. 7 |
| `ucb_vs_ei_bh.ipynb` | `dia12_ucb_vs_ei.ipynb` | `checkpoint_full_grid.pkl`, `all_results_full_grid.csv` (its own local n=10 run; see caveat below) | -- |
| `direct_arylation_benchmark.ipynb` | `dia13_aylation_benchmark.ipynb` | `results_n20_arylation.pkl` | Fig. 6, 14 |
| `snar_rag_grounding.ipynb` | `dia14_grounded_sNAR.ipynb` | `snar_grounding_corpus.json` | (methodology only) |
| `snar_safety_multiagent.ipynb` | `dia15_safety_multiagent_v2.ipynb` | `checkpoint_safety_multiagent_snar.pkl` | Fig. 9-11, 15, Table 2-3 |
| `llm_model_selection_benchmark.ipynb` | `dia8_model_benchmark_v2.ipynb` | `model_benchmark_summary.csv` | Fig. 16 |
| `model_scale_analysis.ipynb` | `dia12_qwen30b_vs_qwen25_analisis.ipynb` | (analysis only, no LLM calls) | -- |

Two of the eight required result files, `seeds_loop_checkpoint.pkl` and
`seeds_loop_checkpoint_run2.pkl`, and one of the code-level matches above
(`all_results_full_grid_20seeds.csv`, actually required by
`scripts/generate_figures.py`) have provenance gaps that predate this
reorganisation; see "Data lineage and reproducibility notes" below.

### Notebook-by-notebook detail

- **`buchwald_hartwig_core_grid.ipynb`**: the 10-seed x 30-iteration core
  grid: GP-BO, SingleAgent, PCV, and three BatchSelect control arms (LLM
  selector / random / top-acquisition), `qwen3:30b`. Also reads
  `seeds_loop_checkpoint.pkl` and `seeds_loop_checkpoint_run2.pkl` as
  comparison baselines (see caveat below). The original notebook additionally
  defined an inline copy of `ejecutar_iteracion_multiagente` that was
  immediately shadowed by a second, equivalent definition later in the same
  notebook, dead code from the moment it was written; that first copy isn't
  reproduced here.

- **`kernel_prior_bh.ipynb`**: KernelPrior / KernelPrior-Loop /
  BatchSelect-RAG vs. GP-BO, 20 seeds x 15 iterations. **Discovered
  inconsistency, preserved deliberately:** this notebook's own LLM-calling
  code defaults to `qwen2.5:7b`, even though sibling files' documentation
  (and the paper) describe this study as run with `qwen3:30b`, see the
  notebook's own markdown cell for detail. I haven't "fixed" this here,
  since doing so would be a genuine behaviour change, not a refactor. The
  original notebook (`dia11_prompt_sensitivity.ipynb`) also contained a large
  amount of dead code: an unused 6-method/10-seed/30-iteration core grid,
  unused UCB batch-proposal functions, and an unused single-agent path, none
  of it ever called by the loop that actually produced the checkpoint above.
  None of it is reproduced here. It also had four near-duplicate sibling
  notebooks (`dia11_bo_kernel_shaped*.ipynb`,
  `dia11_batch_llm_grounded_qwen30b*.ipynb`) that split this same run by
  method or by seed range purely for HPC scheduling; I removed those as
  redundant with this single notebook, per the explicit request that
  prompted this reorganisation.

- **`ucb_vs_ei_bh.ipynb`**: adds the UCB acquisition arm (`bo_puro_ucb`,
  `multiagente_ucb`, `batch_llm_ucb`) alongside EI. **Provenance caveat:**
  this notebook's own local run (10 seeds x 30 iterations) writes
  `checkpoint_full_grid.pkl` / `all_results_full_grid.csv` (n=10), a
  *different, smaller* file from `all_results_full_grid_20seeds.csv` (n=20),
  which is what `scripts/generate_figures.py` actually requires. The
  notebook's own markdown documents that an earlier version of this same
  file, run locally with `qwen2.5:7b` and 20 seeds, produced the n=20 file
  before being edited down to the smaller `qwen3:30b` grid seen here; the
  n=20 file isn't reproducible from the code as currently saved. Both files
  are shipped in this directory regardless.

- **`direct_arylation_benchmark.ipynb`**: GP-BO / PCV / BatchSelect, EI and
  UCB, 20 seeds x 20 iterations. Fetches its dataset directly over HTTP from
  the `edbo` project's GitHub repository (Perera et al. 2018) rather than a
  local file. The original notebook ran this in three incremental batches
  (seeds 0-4, 5-9, then 10-19) purely for session-continuity convenience;
  this version runs all 20 seeds in one loop, which changes nothing about
  the result since each seed is an independent campaign. The three
  intermediate checkpoints from the original incremental run
  (`checkpoint_n5_arylation.pkl`, `checkpoint_n5_9_arylation.pkl`,
  `checkpoint_n10_19_arylation.pkl`, `results_n10_arylation.pkl`) are kept
  for provenance but aren't needed to reproduce `results_n20_arylation.pkl`.

- **`snar_rag_grounding.ipynb`**: builds and leakage-checks the RAG safety
  corpus (3 LibreTexts chapters + 1 Wikipedia article on SNAr, scraped with
  `trafilatura`), and pilots a small no-RAG-vs-RAG-grounded Critic comparison
  (5 seeds) on the SNAr kinetic ODE simulator (Hone et al. 2017) later reused
  by `snar_safety_multiagent.ipynb`.

- **`snar_safety_multiagent.ipynb`**: the paper's safety-constrained case
  study: GP-BO, SingleAgent, SafeAgent, SafePCV (`multiagent_specialists`,
  an iterative safety-gated recheck loop with a dedicated Safety Fallback
  Agent), 20 seeds x 15 iterations, `qwen2.5:7b`. This is `v2` of the
  experiment; `v1` (`dia15_safety_multiagent.ipynb`) explicitly deleted its
  own checkpoint immediately after writing it, in its own code, a
  discarded pilot run, and has been removed from this repository. `v2` is
  the confirmed source of the shipped checkpoint (file timestamp and exact
  row count both match; see the notebook's own markdown for the evidence).

- **`llm_model_selection_benchmark.ipynb`**: benchmarks `qwen2.5:7b` vs.
  `qwen3:14b` latency/throughput/JSON-reliability, the methodology behind
  choosing which local model to run the main experiments with.
  `dia8_model_benchmark.ipynb` (v1) measured an anomalous `qwen3:14b`
  slowdown later diagnosed (in this, the `v2` notebook) as Qwen3's hidden
  "thinking" mode running uncontrolled; v1 has been removed as superseded
  (both wrote to the same output filenames, so only v2's corrected numbers
  ever survived on disk).

- **`model_scale_analysis.ipynb`**: pure analysis (no LLM calls, no BO):
  cross-checks and compares the `qwen2.5:7b` and `qwen3:30b` grids from
  `ucb_vs_ei_bh.ipynb`. Its own seven output figures
  (`fig1_fallback_rate.png` ... `fig7_convergence_curves.png`, loose at the
  top level of this directory) are a *different, smaller* analysis from the
  canonical `figures/fig1..fig16` set in the repository root, same
  numbering, different content, don't confuse the two.

### Removed (see git history of the original project for these, not present here)

- `dia1_python.ipynb` through `dia6_suzuki_benchmark.ipynb`, `dia9.ipynb`:
  early prototyping on a synthetic toy objective and a Suzuki-coupling side
  benchmark, superseded once the real Buchwald-Hartwig pipeline existed.
  `dia9.ipynb` in particular was saved in a broken, non-reproducible state
  (referenced undefined variables) and produced no persisted output.
- `dia7_portfolio_agents.ipynb`: an unrelated financial-portfolio-allocation
  side experiment, off-topic for this paper, and containing a live plaintext
  API key in its source. **If that notebook was ever committed to version
  control or shared elsewhere, that key should be revoked/rotated.**
- `dia8_model_benchmark.ipynb` (v1): superseded by `llm_model_selection_benchmark.ipynb`.
- `dia11_prompt_sensitivity.ipynb`'s four seed/method-split sibling notebooks
  (`dia11_bo_kernel_shaped_qwen30b.ipynb`,
  `dia11_bo_kernel_shaped_ard_loop_qwen30b.ipynb`,
  `dia11_batch_llm_grounded_qwen30b.ipynb` + 4 seed-range variants): redundant
  HPC-scheduling splits of `kernel_prior_bh.ipynb`. Their own smoke-test-only
  checkpoint outputs were removed with them.
- `dia15_safety_multiagent.ipynb` (v1): superseded pilot, self-deletes its
  own checkpoint in its own code.
- `test_ollama_gpu.ipynb`, `test_qwen30b_gpu.ipynb`, `test_nothink_qwen30b.ipynb`:
  infrastructure smoke tests (CUDA/Ollama connectivity), no file I/O, no
  bearing on any result.

## Data lineage and reproducibility notes

Two of the eight required result files, `seeds_loop_checkpoint.pkl` and
`seeds_loop_checkpoint_run2.pkl`, are **not produced by any notebook in
this repository**. `buchwald_hartwig_core_grid.ipynb` reads both as
comparison baselines, and multiple other notebooks refer to "the same
pipeline as dia9", but `dia9.ipynb` (removed, see above) was saved in a
broken, non-reproducible state and never wrote either file. My best guess
is that they were produced by an earlier iteration of that notebook that
was subsequently edited in place without preserving that version. **These
two files are still included here** (they're required inputs to
`scripts/generate_figures.py`, and the validation in `VALIDATION.md`
confirms they reproduce the paper's exact reported numbers). Only the
notebook-level record of exactly how they were originally produced is
incomplete.

`all_results_full_grid_20seeds.csv` (required by `scripts/generate_figures.py`)
and `checkpoint_comparacion_kernel_shaping.pkl` have similar caveats specific
to `ucb_vs_ei_bh.ipynb` and `kernel_prior_bh.ipynb` respectively, see each
notebook's entry above.

None of this affects the validated fast-reproduction path
(`scripts/generate_figures.py` against the shipped result files, see
`VALIDATION.md`), which is what should be used to check the paper's reported
figures and statistics. It only means that a from-scratch re-run of every
notebook top-to-bottom isn't guaranteed to regenerate every file
bit-for-bit; the underlying methodology (architecture, prompts, seeds,
iteration counts) is fully documented in the notebook/package code
regardless.

## Fixes applied (from the original packaging pass, still in effect)

1. **`dia6_suzuki_benchmark.ipynb`** (removed in this pass, but the fix
   remains documented for history): a hardcoded absolute path into a
   specific machine's virtual environment was replaced with a portable
   lookup via the installed `olympus` package.
2. A local Windows username baked into *stored library warning text* (not
   code) in several notebooks was scrubbed to `<user>`.

No other content was modified beyond the class extraction, renaming, and
markdown/comment English-language cleanup described above and in the root
`README.md`.
