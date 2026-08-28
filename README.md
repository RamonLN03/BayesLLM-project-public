# Multi-Agent LLM Expert Systems for Bayesian Experimental Design in Chemistry

Code accompanying the MSc research paper *"Multi-Agent LLM Expert Systems for
Bayesian Experimental Design in Chemistry"* (Ramón López Nieto, Imperial
College London, 2026).

The paper benchmarks five ways of using an LLM agent inside a Gaussian
Process Bayesian Optimisation (GP-BO) loop for reaction-yield optimisation,
against the plain GP-BO baseline, on two high-throughput experimentation
(HTE) datasets (Buchwald-Hartwig C-N coupling, Direct Arylation) and a
mechanistic SNAr kinetic simulator used for a safety-constrained case study.
Two candidate-level architectures let the LLM refine or select the point the
surrogate proposes (**PCV**: Proposer-Critic-Verifier; **BatchSelect**: batch
acquisition with LLM selection, plus a retrieval-grounded variant
**BatchSelect-RAG**); one surrogate-level architecture (**KernelPrior**,
and its iterative variant **KernelPrior-Loop**) lets the LLM set GP kernel
lengthscale priors; and two safety-oriented variants (**SafeAgent**,
**SafePCV**) add retrieval-grounded hazard reasoning. All local LLM
inference uses open-weight Qwen models (`qwen2.5:7b`, `qwen3:30b`) served
locally through [Ollama](https://ollama.com).

## Repository structure

```
.
├── src/bayesllm/            Shared library: the GP-BO / PCV / BatchSelect / KernelPrior
│                            logic that was duplicated across several notebooks, factored
│                            into BuchwaldHartwigBenchmark and DirectArylationBenchmark
│                            classes. See experiments/README.md for what was and wasn't
│                            extracted, and why.
├── experiments/            Jupyter notebooks (the "reference implementation") plus the
│                            checkpoint/result files they read and write, and the raw
│                            benchmark dataset. See experiments/README.md.
├── scripts/                 Deterministic figure/table generation (no LLM, no GPU, no BO —
│                            reads the result files in experiments/, writes figures/ and
│                            figures_supplementary/). See "Reproducing the paper" below.
├── figures/                 Canonical output of scripts/generate_figures.py — every figure
│                            and statistical-test table cited in section4_results.tex.
├── figures_supplementary/  Output of scripts/generate_figures_supplementary.py and
│                            scripts/generate_figure3_hero.py — an alternative, more heavily
│                            styled pass over the same results (likely source of the polished
│                            main-body "hero" figures; see experiments/README.md for caveats).
├── hpc/                     PBS job scripts used to run the notebooks on Imperial College's
│                            HPC cluster (GPU nodes, Ollama-served Qwen models).
├── pyproject.toml           Makes `bayesllm` an installable package (`pip install -e .`).
├── requirements.txt
├── VALIDATION.md            What was checked to confirm this reorganised repository
│                            reproduces the original results, and how to re-check it yourself.
└── LICENSE                  MIT
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.13 was used to produce the reported results. RDKit and BoTorch wheels
are readily available for 3.10–3.13 on Windows/Linux/macOS. The `pip install -e .`
step installs the `bayesllm` package (`src/bayesllm/`) in editable mode, which
`scripts/generate_figures.py` does not need but every notebook in `experiments/`
that runs a live campaign does.

## Reproducing the paper's figures and tables (fast path, no GPU needed)

Every figure and table cited in the paper's Results section is produced from
a small set of pre-computed result files already included in
`experiments/` (the outputs of the notebooks described in
`experiments/README.md`). Reproducing them from those files takes seconds:

```bash
python scripts/generate_figures.py                 # -> figures/  (fig1..fig16, table1..table3)
python scripts/generate_figures_supplementary.py    # -> figures_supplementary/
python scripts/generate_figure3_hero.py             # -> figures_supplementary/fig3_bh_convergence.pdf
```

`scripts/generate_figures.py` is the canonical one: its output filenames are
exactly the ones referenced by `\includegraphics` in the paper's LaTeX
source. See [VALIDATION.md](VALIDATION.md) for the check confirming this
reorganised, path-portable version of the script reproduces the original
byte-for-byte (all 7 statistical tables) or pixel-for-pixel (34 of 38
figures; the remaining 4 differ only in unseeded scatter-plot jitter that
was already non-deterministic in the original code).

## Reproducing the underlying Bayesian optimisation campaigns (slow path)

Regenerating the result files themselves means re-running the optimisation
campaigns in `experiments/*.ipynb`, which call a locally served LLM at every
iteration. This is expensive: the paper's experimental grid (see the
"Experimental Protocol" table in the paper) spans up to 20 seeds × 30
iterations × 6 architectures per benchmark, and individual HPC jobs in
`hpc/` were budgeted 4–72 hours of GPU walltime. You will need:

- [Ollama](https://ollama.com) installed and the `qwen2.5:7b` and/or
  `qwen3:30b` models pulled (`ollama pull qwen2.5:7b`) — each is several GB.
- A CUDA GPU for practical runtimes (the original runs used a single L40S GPU
  via Imperial College's HPC/PBS scheduler).
- The Python environment above (including `pip install -e .`), active in the
  same working directory as the notebook (each notebook reads/writes files by
  a path relative to its own location — see `experiments/README.md`).

`hpc/` contains the original PBS job scripts (Imperial College RCS
scheduler) used to run each notebook unattended via `jupyter nbconvert
--execute`; see `hpc/README.md`. Running a notebook directly in Jupyter
works the same way, just without the HPC scheduling.

**Honesty note:** not every notebook's saved state is confirmed to
reproduce its shipped result file byte-for-byte from a fresh top-to-bottom
run — some were edited after the run that actually produced the currently
shipped `.pkl`/`.csv`, and two result files have no notebook in this
repository that writes them. This is documented in detail, file by file, in
["Data lineage and reproducibility notes"](experiments/README.md#data-lineage-and-reproducibility-notes).
It does not affect the fast path above, which uses the shipped result files
directly.

## Required dataset

`experiments/data/Dreher_and_Doyle_input_data.xlsx` is the Buchwald-Hartwig
C-N coupling HTE dataset of Ahneman et al. (2018), *Predicting reaction
performance in C-N cross-coupling using machine learning*, Science 360,
186-190 — 3,955 measured yields over a 4-component reaction space (ligand,
additive, base, aryl halide), originally released by Dreher & Doyle. It is
included in this repository (2.1 MB) for convenience; see
`experiments/data/README.md` for full provenance and license notes.

The Direct Arylation benchmark (Perera et al. 2018, *A platform for
automated nanomole-scale reaction screening and micromole-scale synthesis in
flow*, Science 359, 429-434) is fetched at run time directly from its public
CSV mirror in the `edbo` project's GitHub repository (see
`bayesllm.direct_arylation`) rather than vendored here.

## Citation

If you use this code, please cite the paper:

```
López Nieto, R. (2026). Multi-Agent LLM Expert Systems for Bayesian
Experimental Design in Chemistry. MSc thesis, Imperial College London.
```

## License

MIT — see [LICENSE](LICENSE). The bundled dataset
(`experiments/data/Dreher_and_Doyle_input_data.xlsx`) retains its own
original terms; see `experiments/data/README.md`.
