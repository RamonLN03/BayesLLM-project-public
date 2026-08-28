# HPC job scripts

PBS job scripts used to run the notebooks in `experiments/` unattended on
Imperial College London's Research Computing Service (RCS) HPC cluster, on a
single GPU node (`gpu_type=L40S` where pinned) with a locally served Ollama
model.

Each script follows the same pattern:

```bash
cd $HOME/bayesllm-project/experiments   # working directory the notebook's relative I/O expects
module load Python/3.13.5-GCCcore-14.3.0
source venv/bin/activate
export PATH=$HOME/ollama/bin:$PATH

ollama serve &                          # start a local Ollama server for this job
sleep 10
ollama pull qwen3:30b                   # (where applicable)

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.allow_errors=True \
  <notebook>.ipynb
```

`--ExecutePreprocessor.allow_errors=True` lets a run continue past a failed
cell rather than aborting the whole (multi-hour) job; check the executed
notebook for cells with `"output_type": "error"` afterwards rather than
assuming a clean exit means every cell succeeded.

The changes made to these scripts for this repository are: the `cd`
target, updated from `$HOME/bayesllm-project` to
`$HOME/bayesllm-project/experiments` to match this repository's layout
(notebooks now live in `experiments/`, not the repository root); the
notebook filename each script executes, updated to match the renamed
notebook it runs (e.g. `dia10.ipynb` → `buchwald_hartwig_core_grid.ipynb`);
and the file itself renamed to match. No PBS resource request, walltime,
queue, or execution logic was changed. Five scripts that ran notebooks
since removed as redundant (four HPC seed/method splits of
`kernel_prior_bh.ipynb`'s run, plus a smoke-test-only job) were removed
along with them — see `experiments/README.md` for why those notebooks were
removed.

| Script | Runs | Was |
|---|---|---|
| `buchwald_hartwig_core_grid.pbs` | `buchwald_hartwig_core_grid.ipynb` | `dia10_full.pbs` |
| `kernel_prior_bh.pbs` | `kernel_prior_bh.ipynb` | `dia11_full.pbs` |
| `ucb_vs_ei_bh.pbs` | `ucb_vs_ei_bh.ipynb` | `dia12_full.pbs` |
| `direct_arylation_benchmark.pbs` | `direct_arylation_benchmark.ipynb` | `dia13_full.pbs` |
| `snar_rag_grounding.pbs` | `snar_rag_grounding.ipynb` | `dia14_full.pbs` |
| `test_internet.pbs` | (connectivity check, no notebook) | unchanged |

`snar_safety_multiagent.ipynb` and `llm_model_selection_benchmark.ipynb` had
no dedicated `_full.pbs` script in the original project (run some other way
— interactively, or via a script not captured in this repository).

## Running a notebook without a scheduler

Outside an HPC/PBS environment, the same effect is achieved by starting
`ollama serve` locally, pulling the required model, and either running the
notebook interactively in Jupyter or via:

```bash
jupyter nbconvert --to notebook --execute --inplace experiments/<notebook>.ipynb
```

from the repository root — the important part is that the notebook's
working directory is `experiments/`, since its file reads/writes
(checkpoints, the dataset in `experiments/data/`) are relative to it.

## Reproducibility caveat

Several of these jobs are documented (in the `.pbs` file comments
themselves) as having failed or been superseded during the actual research
process — e.g. a `NameError` from a missing `import time`, or an
`OLLAMA_HOST` port collision between concurrently running jobs on the same
node. See `experiments/README.md#data-lineage-and-reproducibility-notes` for
which result files this affects.
