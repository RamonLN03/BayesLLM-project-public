# Dataset

**`Dreher_and_Doyle_input_data.xlsx`** — the Buchwald-Hartwig C-N
cross-coupling high-throughput experimentation (HTE) dataset used as the
primary benchmark in the paper.

- **Source**: Ahneman, D. T., Estrada, J. G., Lin, S., Dreher, S. D., &
  Doyle, A. G. (2018). *Predicting reaction performance in C-N
  cross-coupling using machine learning*. Science, 360(6385), 186-190.
  https://doi.org/10.1126/science.aar5169
- **Content**: 3,955 measured yields for palladium-catalysed C-N
  cross-coupling reactions, varying 4 categorical components (ligand: 4
  options, additive: 22, base: 3, aryl halide: 15) across multiple plates
  and cross-validation folds (sheets `FullCV_01`–`FullCV_10`, `Test1`–`Test4`,
  `Plates1-3`, `Plate2_new`).
- **Used by**: `bayesllm.buchwald_hartwig.BuchwaldHartwigBenchmark`, and
  through it every Buchwald-Hartwig notebook
  (`buchwald_hartwig_core_grid.ipynb`, `kernel_prior_bh.ipynb`,
  `ucb_vs_ei_bh.ipynb`) — read via `pandas.read_excel`, sheet `FullCV_01`.
  Eight RDKit-computed physicochemical descriptors per component
  (molecular weight, TPSA, MolLogP, H-bond donors/acceptors, rotatable
  bonds, max/min Gasteiger partial charge) are used to cast the categorical
  reaction space as a continuous 32-dimensional input space for the GP
  surrogate; a `RandomForestRegressor` (300 estimators, trained once and
  held fixed) serves as the emulator that Bayesian Optimisation queries.

This file is bundled in the repository (2.1 MB) for convenience. It
originates from the public data release accompanying the Ahneman et al.
2018 Science paper; consult that publication/its supplementary materials for
the dataset's original license terms before reuse outside this repository.

The second benchmark used in the paper (Direct Arylation, Perera et al.
2018) is **not** vendored here — it is fetched at run time directly from a
public CSV mirror (see `bayesllm.direct_arylation.DirectArylationBenchmark`),
so no local copy is needed.
