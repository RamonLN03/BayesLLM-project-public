"""Buchwald-Hartwig C-N cross-coupling benchmark, and the LLM agent
architectures benchmarked against it (PCV, BatchSelect, SingleAgent,
KernelPrior).

This module is a behaviour-preserving extraction of the 28 functions that
were verified byte-for-byte identical (by programmatic diff, before any
extraction) across ``dia10.ipynb``, ``dia11_prompt_sensitivity.ipynb`` and
``dia12_ucb_vs_ei.ipynb`` in the original project. Every method below is the
corresponding original function with only its former module-level globals
(``scaler``, ``FEATURE_COLUMNS``, ``objetivo``, ``BOUNDS``, ``call_llm``,
...) replaced by ``self`` attributes/methods -- no algorithmic change.

The one confirmed behavioural difference between the original notebooks was
``call_llm``'s default model (``qwen3:30b`` in dia10/dia12,
``qwen2.5:7b`` in dia11_prompt_sensitivity as saved on disk, despite that
notebook's own markdown claiming qwen3:30b -- see
``experiments/README.md`` for this discrepancy). That is exactly what the
``model`` constructor parameter is for: each experiment notebook passes the
value matching what the original notebook actually used.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from .llm import call_llm

DESCRIPTOR_NAMES = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "MaxPartialCharge", "MinPartialCharge",
]


def _compute_descriptors(smiles: str) -> dict:
    """Convert a SMILES string into a dict of physicochemical descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
    }


def _build_descriptor_table(smiles_series, prefix: str) -> pd.DataFrame:
    unique_smiles = smiles_series.unique()
    records = {smi: _compute_descriptors(smi) for smi in unique_smiles}
    lookup = pd.DataFrame.from_dict(records, orient="index")
    lookup.columns = [f"{prefix}_{col}" for col in lookup.columns]
    return lookup


class BuchwaldHartwigBenchmark:
    """RandomForest emulator over the Dreher & Doyle Buchwald-Hartwig HTE
    dataset (3,955 reactions, 32-D RDKit descriptor space), plus every LLM
    agent architecture benchmarked against it: PCV (Proposer-Critic-Verifier),
    BatchSelect (LLM selector over a qLogNEI batch, plain or ARD-grounded),
    SingleAgent, and the KernelPrior ARD-lengthscale-eliciting helpers.

    Parameters
    ----------
    data_path:
        Path to the Dreher & Doyle Excel dataset (sheet ``FullCV_01``).
    model:
        Ollama model name passed to every LLM call made through this
        instance. Must match what the notebook being reproduced actually
        used -- see the module docstring.
    """

    def __init__(self, data_path: str = "data/Dreher_and_Doyle_input_data.xlsx",
                 model: str = "qwen3:30b"):
        self.model = model

        df = pd.read_excel(data_path, sheet_name="FullCV_01")

        ligand_lookup = _build_descriptor_table(df["Ligand"], "ligand")
        additive_lookup = _build_descriptor_table(df["Additive"], "additive")
        base_lookup = _build_descriptor_table(df["Base"], "base")
        arylhalide_lookup = _build_descriptor_table(df["Aryl halide"], "arylhalide")

        features = df[["Ligand", "Additive", "Base", "Aryl halide", "Output"]].copy()
        features = features.merge(ligand_lookup, left_on="Ligand", right_index=True)
        features = features.merge(additive_lookup, left_on="Additive", right_index=True)
        features = features.merge(base_lookup, left_on="Base", right_index=True)
        features = features.merge(arylhalide_lookup, left_on="Aryl halide", right_index=True)

        self.feature_columns = [
            col for col in features.columns
            if col not in ["Ligand", "Additive", "Base", "Aryl halide", "Output"]
        ]
        self.X_raw = features[self.feature_columns].values
        self.y_raw = features["Output"].values

        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_raw)

        self.emulator = RandomForestRegressor(n_estimators=300, random_state=42)
        self.emulator.fit(self.X_scaled, self.y_raw)

        self.n_dims = self.X_scaled.shape[1]
        self.bounds = torch.tensor([[0.0] * self.n_dims, [1.0] * self.n_dims], dtype=torch.float64)

        self.descriptor_bounds_physical = {
            col: (self.X_raw[:, i].min(), self.X_raw[:, i].max())
            for i, col in enumerate(self.feature_columns)
        }

        # Deterministic redundancy check disabled -- see design-fix diagnosis
        # (19 July 2026) in the original dia10 notebook: even after rescaling
        # by sqrt(k/N_DIMS) over adjusted_dims only, fallback stayed ~100% in
        # the smoke test, likely due to a low-dimension projection effect
        # against the dense 64-point initial Sobol design. The Critic relies
        # on the bounds check + the LLM's own semantic judgment instead.
        self.redundancy_threshold = 0.0

    # -- LLM client -----------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        return call_llm(system_prompt, user_prompt, model=self.model)

    # -- Objective and candidate <-> dict conversions --------------------
    def objective(self, X):
        """Emulated black-box objective: X is a tensor (n, 32) of scaled
        molecular descriptors, each dimension in [0, 1]. Returns a tensor
        (n, 1) of predicted reaction yield (%)."""
        X_numpy = X.detach().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_pred = self.emulator.predict(X_numpy)
        return torch.tensor(y_pred, dtype=torch.float64).unsqueeze(-1)

    def dict_to_tensor(self, candidate_dict: dict) -> torch.Tensor:
        """Convert a {feature_name: value} dict into a (1, N_DIMS) tensor,
        respecting the feature_columns order."""
        values = [candidate_dict[col] for col in self.feature_columns]
        return torch.tensor([values], dtype=torch.float64)

    def tensor_to_dict(self, x_tensor: torch.Tensor) -> dict:
        """Convert a (N_DIMS,) or (1, N_DIMS) tensor into a
        {feature_name: value} dict."""
        x_flat = x_tensor.flatten()
        return {col: x_flat[i].item() for i, col in enumerate(self.feature_columns)}

    def to_physical(self, candidate_scaled_dict: dict) -> dict:
        """Convert a {feature_name: value in [0,1]} dict into physical units
        (g/mol, A^2, etc.) via the fitted MinMaxScaler, so the LLM reasons on
        chemically meaningful numbers instead of normalised ones."""
        scaled_array = np.array([[candidate_scaled_dict[col] for col in self.feature_columns]])
        physical_array = self.scaler.inverse_transform(scaled_array)[0]
        return {col: physical_array[i] for i, col in enumerate(self.feature_columns)}

    def to_scaled(self, candidate_physical_dict: dict) -> dict:
        """Inverse of to_physical: convert physical-unit values back into the
        [0,1] normalised space that BO/the emulator operate in. Clips to
        [0,1] in case the LLM proposes a value outside the range seen during
        fitting."""
        physical_array = np.array([[candidate_physical_dict[col] for col in self.feature_columns]])
        scaled_array = self.scaler.transform(physical_array)[0]
        scaled_array = np.clip(scaled_array, 0.0, 1.0)
        return {col: scaled_array[i] for i, col in enumerate(self.feature_columns)}

    def format_candidate(self, candidate_scaled_dict: dict) -> str:
        """Render a full 32D candidate as readable, physically meaningful
        text for a prompt (converts from normalised [0,1] to physical units
        first)."""
        physical = self.to_physical(candidate_scaled_dict)
        return ", ".join(f"{name}={value:.3f}" for name, value in physical.items())

    def format_history(self, historial: list) -> str:
        """Render the experiment history compactly: iteration/yield trend
        for all past experiments, plus the full descriptor vector (in
        physical units) of the best one found so far."""
        lines = ["Yield trend so far (iteration: yield):"]
        for i, exp in enumerate(historial):
            lines.append(f"  {i + 1}: {exp['yield']:.2f}%")

        best_exp = max(historial, key=lambda exp: exp['yield'])
        best_idx = historial.index(best_exp) + 1
        lines.append(f"\nBest experiment so far (iteration {best_idx}, yield={best_exp['yield']:.2f}%):")
        lines.append("  " + self.format_candidate(best_exp['descriptors']))

        return "\n".join(lines)

    def format_descriptor_bounds(self) -> str:
        """Render each descriptor's observed physical min/max range as
        compact text, so the Proposer/Verifier have the information they
        need to avoid proposing values outside the real dataset's range."""
        lines = [f"{col}: [{lo:.3f}, {hi:.3f}]" for col, (lo, hi) in self.descriptor_bounds_physical.items()]
        return "\n".join(lines)

    # -- Prompts ----------------------------------------------------------
    def _build_proposer_prompt(self) -> str:
        return f"""You are an expert chemist acting as a creative experimentalist in a
sequential Bayesian optimisation campaign for a Buchwald-Hartwig C-N cross-coupling
reaction. You receive a candidate experiment proposed by Bayesian Optimisation,
described by 32 physicochemical descriptors (in physical units) of its four
components (ligand, additive, base, aryl halide), plus the experiment history.

You do NOT need to modify all 32 dimensions. Only propose adjustments to the
dimensions you judge chemically relevant given your reasoning (e.g. partial
charge descriptors as a proxy for electronic effects, or TPSA/LogP as a proxy
for polarity and sterics). Leave everything else unchanged, if you have no
strong chemical justification for changing a dimension, do not include it.

Any value you propose for a dimension MUST stay within the physically
observed range for that descriptor in the real dataset, shown below:
{self.format_descriptor_bounds()}

Respond ONLY with a valid JSON object, with "adjustments" decided FIRST and "reasoning"
written to describe exactly those adjustments:
{{
    "adjustments": [
        {{"dimension": "<one of the descriptor names shown>", "new_value": <number, physical units>}}
    ],
    "reasoning": "<string explaining your chemical rationale for exactly the adjustments
above -- any value you mention here MUST match the corresponding new_value exactly>"
}}
"adjustments" may be an empty list if you approve the candidate as-is.
No additional text, no markdown."""

    def _build_critic_prompt(self) -> str:
        return """You are a rigorous chemistry expert reviewing an experimental proposal
for a Buchwald-Hartwig C-N cross-coupling reaction. The proposal has ALREADY
passed automated bounds and redundancy checks (not your concern). Your ONLY job
is to evaluate whether the chemical reasoning provided is coherent, plausible,
and consistent with the experimental history shown. Reject only if the reasoning
is contradictory, nonsensical, or unsupported by the data shown.

Respond ONLY with a valid JSON object:
{
    "approved": <true or false>,
    "rejection_reason": "<string if rejected, else null>",
    "concern_level": "<low|medium|high>"
}
No additional text, no markdown."""

    def _build_verifier_prompt(self) -> str:
        return f"""You are the lead scientist overseeing a Buchwald-Hartwig cross-coupling
optimisation campaign. You receive a proposal that has passed safety and
coherence review. Make the final decision: confirm it as-is, or make a minor
adjustment to a small number of dimensions if you have strong chemical grounds
to do so. Log your reasoning for interpretability.

You do NOT need to modify all 32 dimensions, only propose adjustments where
you have a clear chemical justification.

Any value you propose for a dimension MUST stay within the physically
observed range for that descriptor in the real dataset, shown below:
{self.format_descriptor_bounds()}

Respond ONLY with a valid JSON object:
{{
    "adjustment_made": <true or false>,
    "adjustments": [
        {{"dimension": "<one of the descriptor names shown>", "new_value": <number, physical units>}}
    ],
    "final_reasoning": "<string: why this experiment is worth running>"
}}
"adjustments" may be an empty list if you confirm the proposal as-is.
No additional text, no markdown."""

    def _build_single_agent_prompt(self) -> str:
        return f"""You are an expert chemist with a long and successful research career.
You are running a sequential Bayesian optimisation campaign for a Buchwald-Hartwig
C-N cross-coupling reaction, described by 32 physicochemical descriptors of its four
components. Bayesian Optimisation does not account for chemical safety, feasibility,
or plausibility. Your job is to review the point it proposes and either approve it
as-is or make targeted adjustments based on your chemical expertise.

You do NOT need to modify all 32 dimensions -- only propose adjustments to the
dimensions you have a clear chemical justification to change.

Any value you propose for a dimension MUST stay within the physically
observed range for that descriptor in the real dataset, shown below:
{self.format_descriptor_bounds()}

Respond ONLY with a valid JSON object, deciding "adjustments" FIRST and writing "reasoning"
to describe exactly those adjustments:
{{
    "approved": <true or false>,
    "adjustments": [
        {{"dimension": "<one of the descriptor names shown>", "new_value": <number, physical units>}}
    ],
    "reasoning": "<string explaining your evaluation -- any value you mention here MUST
match the corresponding new_value exactly>"
}}
"adjustments" may be an empty list if you approve the candidate as-is.
No additional text, no markdown."""

    def _build_selector_prompt(self) -> str:
        return """You are an expert chemist selecting which candidate experiment to run
next from a batch proposed by Bayesian Optimisation, for a Buchwald-Hartwig C-N
cross-coupling reaction. Each candidate is described by 32 physicochemical
descriptors of its four components. The candidates are shown in NO particular
order -- do not assume the order reflects any ranking or preference. Use your
own chemical reasoning, together with the experiment history, to judge which
candidate is most promising.

Respond ONLY with a valid JSON object:
{
    "selected_index": <integer, the "Candidate N" index of your choice>,
    "reasoning": "<string explaining your choice>"
}
No additional text, no markdown."""

    def _build_selector_prompt_grounded(self, ard_importance_text: str) -> str:
        return f"""You are an expert chemist selecting which candidate experiment to run
next from a batch proposed by Bayesian Optimisation, for a Buchwald-Hartwig C-N
cross-coupling reaction. Each candidate is described by 32 physicochemical
descriptors of its four components. The candidates are shown in NO particular
order -- do not assume the order reflects any ranking or preference. Use your
own chemical reasoning, together with the experiment history, to judge which
candidate is most promising.

A Gaussian Process model has been fitted to the experiment history so far. Based
on its learned lengthscales, the dimensions below are CURRENTLY the most
influential on predicted yield (shortest lengthscale = most influential; this
ranking updates as more data is collected and may not match textbook chemical
intuition):
{ard_importance_text}

Prioritise these dimensions in your reasoning where relevant, alongside your own
chemical judgement.

Respond ONLY with a valid JSON object:
{{
    "selected_index": <integer, the "Candidate N" index of your choice>,
    "reasoning": "<string explaining your choice>"
}}
No additional text, no markdown."""

    # -- Deterministic checks ---------------------------------------------
    def apply_adjustments(self, base_candidate_scaled: dict, adjustments: list):
        physical = self.to_physical(base_candidate_scaled)
        applied_dims = []
        for adj in adjustments:
            dim = adj.get('dimension')
            value = adj.get('new_value')
            if dim in self.feature_columns and isinstance(value, (int, float)):
                physical[dim] = value
                applied_dims.append(dim)
        physical_array = np.array([[physical[col] for col in self.feature_columns]])
        scaled_array = self.scaler.transform(physical_array)[0]
        out_of_range_dims = [self.feature_columns[i] for i, v in enumerate(scaled_array) if v < 0.0 or v > 1.0]
        scaled_array_clipped = np.clip(scaled_array, 0.0, 1.0)
        candidate_scaled = {col: scaled_array_clipped[i] for i, col in enumerate(self.feature_columns)}
        return candidate_scaled, out_of_range_dims, applied_dims

    def check_bounds(self, out_of_range_dims: list):
        if out_of_range_dims:
            return False, f"Adjustment(s) outside observed physical range: {', '.join(out_of_range_dims)}"
        return True, None

    def check_redundancy(self, candidate_scaled: dict, historial: list, adjusted_dims: list,
                          threshold: Optional[float] = None):
        if threshold is None:
            threshold = self.redundancy_threshold
        k = len(adjusted_dims)
        if k == 0:
            return True, None
        umbral = threshold * (k / self.n_dims) ** 0.5
        x_nuevo = torch.tensor([candidate_scaled[d] for d in adjusted_dims], dtype=torch.float64)
        for exp in historial:
            x_previo = torch.tensor([exp['descriptors'][d] for d in adjusted_dims], dtype=torch.float64)
            distancia = torch.norm(x_nuevo - x_previo).item()
            if distancia < umbral:
                return False, f"Too similar to a previous experiment on adjusted dims {adjusted_dims} (dist={distancia:.3f}, threshold={umbral:.3f})"
        return True, None

    # -- Agent calls --------------------------------------------------------
    def call_proposer(self, candidate_scaled: dict, historial: list) -> dict:
        system_prompt = self._build_proposer_prompt()
        user_prompt = (
            f"{self.format_history(historial)}\n\n"
            f"Bayesian Optimisation proposes the following candidate "
            f"(physical units):\n{self.format_candidate(candidate_scaled)}\n\n"
            f"Enrich this proposal with your chemical reasoning."
        )
        respuesta = self._call_llm(system_prompt, user_prompt)

        adjustments = respuesta.get('adjustments', [])
        merged_scaled, out_of_range_dims, applied_dims = self.apply_adjustments(candidate_scaled, adjustments)

        return {
            'descriptors': merged_scaled,
            'reasoning': respuesta.get('reasoning', ''),
            'out_of_range_dims': out_of_range_dims,
            'adjusted_dims': applied_dims,
        }

    def call_critic(self, propuesta: dict, historial: list) -> dict:
        bounds_ok, razon_bounds = self.check_bounds(propuesta['out_of_range_dims'])
        if not bounds_ok:
            return {
                'approved': False,
                'rejection_reason': razon_bounds,
                'concern_level': 'high',
                'capa_rechazo': 'deterministic',
            }

        redundancia_ok, razon_redundancia = self.check_redundancy(
            propuesta['descriptors'], historial, propuesta['adjusted_dims']
        )
        if not redundancia_ok:
            return {
                'approved': False,
                'rejection_reason': razon_redundancia,
                'concern_level': 'medium',
                'capa_rechazo': 'deterministic',
            }

        system_prompt = self._build_critic_prompt()
        user_prompt = (
            f"{self.format_history(historial)}\n\n"
            f"Proposed experiment (physical units):\n{self.format_candidate(propuesta['descriptors'])}\n\n"
            f"Reasoning given: {propuesta['reasoning']}\n"
            f"Evaluate the coherence of this reasoning."
        )
        veredicto = self._call_llm(system_prompt, user_prompt)
        veredicto['capa_rechazo'] = 'llm' if not veredicto.get('approved', False) else None
        return veredicto

    def call_verifier(self, propuesta: dict, historial: list) -> dict:
        system_prompt = self._build_verifier_prompt()
        user_prompt = (
            f"{self.format_history(historial)}\n\n"
            f"Approved proposal (physical units):\n{self.format_candidate(propuesta['descriptors'])}\n\n"
            f"Original reasoning: {propuesta['reasoning']}\n"
            f"Make your final decision."
        )
        respuesta = self._call_llm(system_prompt, user_prompt)

        adjustments = respuesta.get('adjustments', [])
        final_scaled, out_of_range_dims, applied_dims = self.apply_adjustments(propuesta['descriptors'], adjustments)

        return {
            'descriptors': final_scaled,
            'final_reasoning': respuesta.get('final_reasoning', ''),
            'adjustment_made': respuesta.get('adjustment_made', False),
            'out_of_range_dims': out_of_range_dims,
            'adjusted_dims': applied_dims,
        }

    def call_single_agent(self, candidate_scaled: dict, historial: list) -> dict:
        """Single LLM call that reviews and optionally adjusts BO's candidate
        directly -- no separate Proposer/Critic/Verifier roles, no
        deterministic bounds/redundancy gate."""
        system_prompt = self._build_single_agent_prompt()
        user_prompt = (
            f"{self.format_history(historial)}\n\n"
            f"Bayesian Optimisation proposes the following candidate "
            f"(physical units):\n{self.format_candidate(candidate_scaled)}\n\n"
            f"Evaluate it."
        )
        respuesta = self._call_llm(system_prompt, user_prompt)

        adjustments = respuesta.get('adjustments', [])
        merged_scaled, out_of_range_dims, _ = self.apply_adjustments(candidate_scaled, adjustments)

        return {
            'descriptors': merged_scaled,
            'reasoning': respuesta.get('reasoning', ''),
            'out_of_range_dims': out_of_range_dims,
        }

    def call_selector(self, candidatos_scaled_list: list, historial: list, seed: Optional[int] = None):
        """Ask the Selector agent to pick one candidate from a batch using
        chemical reasoning. The candidates are shuffled before being shown,
        and no acquisition value/ranking is ever included in the prompt."""
        n = len(candidatos_scaled_list)
        rng = np.random.default_rng(seed)
        orden_barajado = rng.permutation(n)

        candidatos_texto = "\n".join(
            f"Candidate {display_i}: {self.format_candidate(candidatos_scaled_list[real_i])}"
            for display_i, real_i in enumerate(orden_barajado)
        )

        system_prompt = self._build_selector_prompt()
        user_prompt = (
            f"{self.format_history(historial)}\n\n"
            f"Batch of candidates proposed by Bayesian Optimisation "
            f"(physical units, unordered):\n{candidatos_texto}\n\n"
            f"Select the most promising candidate."
        )
        respuesta = self._call_llm(system_prompt, user_prompt)

        selected_display_idx = respuesta.get('selected_index', 0)
        if not isinstance(selected_display_idx, int) or not (0 <= selected_display_idx < n):
            selected_display_idx = 0

        selected_real_idx = orden_barajado[selected_display_idx]
        return selected_real_idx, respuesta.get('reasoning', '')

    def call_selector_grounded(self, candidatos_scaled_list: list, historial: list, gp, seed: Optional[int] = None):
        """Same as call_selector, but uses the ARD-grounded prompt variant."""
        n = len(candidatos_scaled_list)
        rng = np.random.default_rng(seed)
        orden_barajado = rng.permutation(n)

        candidatos_texto = "\n".join(
            f"Candidate {display_i}: {self.format_candidate(candidatos_scaled_list[real_i])}"
            for display_i, real_i in enumerate(orden_barajado)
        )

        ard_text = self.format_ard_importance(gp)
        system_prompt = self._build_selector_prompt_grounded(ard_text)
        user_prompt = (
            f"{self.format_history(historial)}\n\n"
            f"Batch of candidates proposed by Bayesian Optimisation "
            f"(physical units, unordered):\n{candidatos_texto}\n\n"
            f"Select the most promising candidate."
        )
        respuesta = self._call_llm(system_prompt, user_prompt)

        selected_display_idx = respuesta.get('selected_index', 0)
        if not isinstance(selected_display_idx, int) or not (0 <= selected_display_idx < n):
            selected_display_idx = 0

        selected_real_idx = orden_barajado[selected_display_idx]
        return selected_real_idx, respuesta.get('reasoning', '')

    # -- GP-BO candidate proposal -----------------------------------------
    def propose_candidate(self, X_bo, Y_bo, bounds=None, seed: Optional[int] = None):
        """Fit the GP and optimize LogEI to propose a single candidate."""
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)

        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq_fn = LogExpectedImprovement(model=gp, best_f=Y_bo.max())
        candidato, _ = optimize_acqf(
            acq_function=acq_fn, bounds=bounds, q=1, num_restarts=10, raw_samples=100
        )
        return candidato

    def propose_batch(self, X_bo, Y_bo, bounds=None, q: int = 4, seed: Optional[int] = None,
                       return_gp: bool = False):
        """Fit the GP and jointly optimize a batch (q) qLogNEI acquisition
        function, returning q candidates proposed together. ``return_gp``
        lets a caller that needs the fitted GP (e.g. to extract ARD
        importance) avoid re-fitting it a second time."""
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)

        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq_fn = qLogNoisyExpectedImprovement(model=gp, X_baseline=X_bo)
        candidatos, _ = optimize_acqf(
            acq_function=acq_fn, bounds=bounds, q=q, num_restarts=10, raw_samples=100
        )
        if return_gp:
            return candidatos, gp
        return candidatos

    def propose_candidate_ucb(self, X_bo, Y_bo, bounds=None, beta: float = 2.0, seed: Optional[int] = None):
        """Same as propose_candidate, but optimizing Upper Confidence Bound
        (fixed beta, as in Srinivas et al. 2010's simplification) instead of
        LogEI, to propose a single candidate."""
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)

        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq_fn = UpperConfidenceBound(model=gp, beta=beta)
        candidato, _ = optimize_acqf(
            acq_function=acq_fn, bounds=bounds, q=1, num_restarts=10, raw_samples=100
        )
        return candidato

    def propose_batch_ucb(self, X_bo, Y_bo, bounds=None, q: int = 4, beta: float = 2.0,
                           seed: Optional[int] = None, return_gp: bool = False):
        """Same batch proposal mechanism as propose_batch, but using a
        q-batch Upper Confidence Bound acquisition function instead of
        qLogNoisyExpectedImprovement."""
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)

        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq_fn = qUpperConfidenceBound(model=gp, beta=beta)
        candidatos, _ = optimize_acqf(
            acq_function=acq_fn, bounds=bounds, q=q, num_restarts=10, raw_samples=100
        )
        if return_gp:
            return candidatos, gp
        return candidatos

    def compute_acquisition_values(self, candidatos, X_bo, Y_bo, bounds=None):
        """Compute a per-candidate q=1 LogEI score for each point in a
        batch. Used only for the 'top acquisition value' control condition
        -- never shown to the LLM Selector."""
        if bounds is None:
            bounds = self.bounds
        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_fn = LogExpectedImprovement(model=gp, best_f=Y_bo.max())
        with torch.no_grad():
            valores = acq_fn(candidatos.unsqueeze(1))
        return valores

    def extract_ard_lengthscales(self, gp) -> dict:
        """Extract the per-dimension ARD lengthscales from a fitted
        SingleTaskGP. Handles both BoTorch kernel layouts:
        covar_module = ScaleKernel(base_kernel) (older default) or
        covar_module = the ARD kernel directly (newer default, since output
        scaling now lives in the Standardize outcome transform)."""
        kernel = gp.covar_module
        if hasattr(kernel, 'lengthscale') and kernel.lengthscale is not None:
            lengthscales = kernel.lengthscale.detach().flatten()
        elif hasattr(kernel, 'base_kernel'):
            lengthscales = kernel.base_kernel.lengthscale.detach().flatten()
        else:
            raise AttributeError(f"Could not locate lengthscale on kernel of type {type(kernel)}")
        return {col: lengthscales[i].item() for i, col in enumerate(self.feature_columns)}

    def format_ard_importance(self, gp, top_k: int = 10) -> str:
        """Render the top_k most influential dimensions (shortest
        lengthscale = most influential) as compact ranked text."""
        lengthscales = self.extract_ard_lengthscales(gp)
        ranked = sorted(lengthscales.items(), key=lambda kv: kv[1])
        lines = [f"  {i+1}. {name} (lengthscale={ls:.3f})" for i, (name, ls) in enumerate(ranked[:top_k])]
        return "\n".join(lines)

    # -- Iteration-level orchestration -------------------------------------
    def run_pcv_deliberation(self, candidato_tensor, X_bo, Y_bo, historial, bounds=None,
                              max_rechazos: int = 3, iteracion: int = 0, verbose: bool = False) -> dict:
        """Run Proposer -> Critic (with Constant-Liar retries) -> Verifier
        deliberation starting from a given candidate tensor. Shared by the
        plain PCV method and BatchSelect, which only differ in how the
        starting candidate is generated."""
        if bounds is None:
            bounds = self.bounds
        X_bo_temp = X_bo.clone()
        Y_bo_temp = Y_bo.clone()

        candidato_dict = self.tensor_to_dict(candidato_tensor)

        intentos = 0
        aprobado = False
        propuesta = None
        log_rechazos = []

        while not aprobado and intentos < max_rechazos:
            propuesta = self.call_proposer(candidato_dict, historial)
            veredicto = self.call_critic(propuesta, historial)

            if veredicto['approved']:
                aprobado = True
            else:
                entrada_log = {
                    'intento': intentos,
                    'capa': veredicto.get('capa_rechazo'),
                    'razon': veredicto.get('rejection_reason'),
                }
                log_rechazos.append(entrada_log)
                if verbose:
                    print(f"  Rechazo [{entrada_log['capa']}]: {entrada_log['razon']}")

                # Constant Liar: fantasize the rejected point as already
                # explored with a pessimistic outcome, so the acquisition
                # surface changes shape and the retry actually explores
                # elsewhere. These are local copies -- they must never leak
                # into the caller's real X_bo/Y_bo.
                x_rejected = self.dict_to_tensor(propuesta['descriptors'])
                y_fantasy = Y_bo_temp.min().reshape(1, 1)
                X_bo_temp = torch.cat([X_bo_temp, x_rejected])
                Y_bo_temp = torch.cat([Y_bo_temp, y_fantasy])

                intentos += 1
                nuevo_candidato = self.propose_candidate(
                    X_bo_temp, Y_bo_temp, bounds, seed=iteracion * 100 + intentos
                )
                candidato_dict = self.tensor_to_dict(nuevo_candidato)

        if aprobado:
            decision = self.call_verifier(propuesta, historial)
            x_final = self.dict_to_tensor(decision['descriptors'])
            fuente = 'multiagent'
            reasoning_final = decision['final_reasoning']
        else:
            x_final = self.dict_to_tensor(candidato_dict)
            fuente = 'bo_fallback'
            reasoning_final = 'Fallback: agents did not reach consensus'

        y_final = self.objective(x_final)

        return {
            'x': x_final,
            'y': y_final,
            'fuente': fuente,
            'rechazos': intentos,
            'reasoning': reasoning_final,
            'log_rechazos': log_rechazos,
        }

    def run_bo_only_iteration(self, X_bo, Y_bo, bounds=None, seed: Optional[int] = None) -> dict:
        """One iteration of plain GP-BO -- no LLM agents at all. BO's raw
        candidate is evaluated directly. This is the baseline condition."""
        if bounds is None:
            bounds = self.bounds
        candidato_tensor = self.propose_candidate(X_bo, Y_bo, bounds, seed=seed)
        y_final = self.objective(candidato_tensor)

        return {
            'x': candidato_tensor,
            'y': y_final,
            'fuente': 'bo_puro',
            'rechazos': 0,
            'reasoning': '',
            'log_rechazos': [],
        }

    def run_single_agent_iteration(self, X_bo, Y_bo, historial, bounds=None,
                                    seed: Optional[int] = None) -> dict:
        """One iteration of the single-agent method."""
        if bounds is None:
            bounds = self.bounds
        candidato_tensor = self.propose_candidate(X_bo, Y_bo, bounds, seed=seed)
        candidato_dict = self.tensor_to_dict(candidato_tensor)

        resultado_llm = self.call_single_agent(candidato_dict, historial)
        x_final = self.dict_to_tensor(resultado_llm['descriptors'])
        y_final = self.objective(x_final)

        return {
            'x': x_final,
            'y': y_final,
            'fuente': 'single_agent',
            'rechazos': 0,
            'reasoning': resultado_llm['reasoning'],
            'log_rechazos': [],
        }

    def generate_initial_design(self, seed: int, n_init: Optional[int] = None):
        """Build the shared initial Sobol design for a given seed. History
        entries store descriptors in SCALED [0,1] form (matching X_init);
        physical-unit conversion happens later, at prompt-formatting time."""
        if n_init is None:
            n_init = 2 * self.n_dims  # 64 initial points, low end of the 2d-10d heuristic
        X_init = draw_sobol_samples(bounds=self.bounds, n=n_init, q=1, seed=seed).squeeze(1)
        Y_init = self.objective(X_init)

        history_init = []
        for i in range(X_init.shape[0]):
            candidate_dict = self.tensor_to_dict(X_init[i])
            history_init.append({
                "descriptors": candidate_dict,
                "yield": Y_init[i].item(),
            })

        return X_init, Y_init, history_init

    @staticmethod
    def clone_run_state(X_init, Y_init, history_init):
        """Fresh, independent copies of the initial design for one (method,
        seed) run, so growing X_bo/Y_bo/historial in one method's run never
        leaks into another's."""
        X_bo = X_init.clone()
        Y_bo = Y_init.clone()
        historial = list(history_init)
        return X_bo, Y_bo, historial

    def run_batch_variant_iteration(self, X_bo, Y_bo, historial, bounds=None, q: int = 4,
                                     prompt_variant: str = 'baseline', max_rechazos: int = 3,
                                     iteracion: int = 0, verbose: bool = False) -> dict:
        """
        Batch-selection iteration for the prompt-sensitivity study. Supports
        two Selector prompt variants: 'baseline' (plain call_selector) or
        'grounded' (ARD-lengthscale importance injected via
        call_selector_grounded). Always uses the LLM selector.
        """
        if bounds is None:
            bounds = self.bounds
        if prompt_variant == 'grounded':
            candidatos_tensor, gp = self.propose_batch(X_bo, Y_bo, bounds, q=q, seed=iteracion, return_gp=True)
            candidatos_dict_list = [self.tensor_to_dict(candidatos_tensor[i]) for i in range(q)]
            idx, selection_reasoning = self.call_selector_grounded(candidatos_dict_list, historial, gp, seed=iteracion)
        elif prompt_variant == 'baseline':
            candidatos_tensor = self.propose_batch(X_bo, Y_bo, bounds, q=q, seed=iteracion)
            candidatos_dict_list = [self.tensor_to_dict(candidatos_tensor[i]) for i in range(q)]
            idx, selection_reasoning = self.call_selector(candidatos_dict_list, historial, seed=iteracion)
        else:
            raise ValueError(f"Unknown prompt_variant: {prompt_variant}")

        candidato_seleccionado = candidatos_tensor[idx].unsqueeze(0)

        resultado = self.run_pcv_deliberation(
            candidato_seleccionado, X_bo, Y_bo, historial,
            bounds=bounds, max_rechazos=max_rechazos, iteracion=iteracion, verbose=verbose
        )
        resultado['fuente'] = f"batch_llm_{prompt_variant}"
        resultado['selection_reasoning'] = selection_reasoning
        resultado['batch_index_selected'] = idx

        return resultado

    # -- KernelPrior: LLM-elicited ARD lengthscale priors -------------------
    KERNEL_WEIGHT_MIN = 0.3
    KERNEL_WEIGHT_MAX = 3.0
    KERNEL_DESCRIPTOR_TYPES = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
                               "NumRotatableBonds", "MaxPartialCharge", "MinPartialCharge"]
    KERNEL_COMPONENT_NAMES = ["ligand", "additive", "base", "arylhalide"]
    KERNEL_ARD_REELICIT_EVERY = 5  # bo_kernel_shaped_ard_loop re-elicits every 5 iterations

    def _build_kernel_shaper_prompt(self) -> str:
        return f"""You are an expert chemist setting priors for a Gaussian Process surrogate
model in a Bayesian optimisation campaign for a Buchwald-Hartwig C-N cross-coupling
reaction. The GP uses one independent lengthscale per input dimension (automatic
relevance determination, ARD): a SHORT lengthscale means the GP treats that
dimension as highly informative (small changes strongly affect the predicted
yield), a LONG lengthscale means the GP treats it as close to irrelevant.

The 32 input dimensions come from computing 8 descriptor TYPES separately for
each of the 4 reaction components (ligand, additive, base, aryl halide). The 8
descriptor types are:
{", ".join(self.KERNEL_DESCRIPTOR_TYPES)}

For each descriptor type you have a genuine chemical opinion about, give a BASE
importance weight that applies to all four components by default. Values above
1.0 mean the GP should treat that descriptor as more informative, values below
1.0 as less informative. If you have a specific mechanistic reason to believe a
descriptor works differently depending on WHICH component it is measured on --
for example, partial charge might matter for the aryl halide's leaving group but
not for the base -- add a component-specific override instead of relying on the
base weight for that component.

Weights must be in the range [{self.KERNEL_WEIGHT_MIN}, {self.KERNEL_WEIGHT_MAX}]. Any
descriptor type you do not mention keeps the neutral weight of 1.0 for all four
components.

Respond ONLY with a valid JSON object:
{{

    "importance": [
        {{
            "descriptor_type": "<one of the 8 types above>",
            "weight": <base weight, applies to all 4 components unless overridden>,
            "overrides": [
                {{"component": "<ligand|additive|base|arylhalide>", "weight": <number>}}
            ]
        }}
    ],
    "reasoning": "<string explaining your chemical rationale>"

}}
"overrides" may be an empty list when the base weight applies uniformly.
No additional text, no markdown."""

    def parse_kernel_weights(self, respuesta: dict) -> dict:
        pesos = {col: 1.0 for col in self.feature_columns}
        for item in respuesta.get('importance', []):
            tipo = item.get('descriptor_type')
            peso_base = item.get('weight')
            if not isinstance(peso_base, (int, float)):
                continue
            peso_base = float(min(self.KERNEL_WEIGHT_MAX, max(self.KERNEL_WEIGHT_MIN, peso_base)))

            if tipo in self.feature_columns:
                # The LLM sometimes returns the full column name (e.g.
                # "arylhalide_MolWt", copying the format it sees in
                # format_ard_importance) instead of the generic 8-descriptor
                # type. Treated as a weight specific to that column.
                pesos[tipo] = peso_base
                continue

            if tipo not in self.KERNEL_DESCRIPTOR_TYPES:
                continue

            for comp in self.KERNEL_COMPONENT_NAMES:
                pesos[f"{comp}_{tipo}"] = peso_base
            for ov in item.get('overrides', []):
                comp = ov.get('component')
                w = ov.get('weight')
                col = f"{comp}_{tipo}"
                if comp in self.KERNEL_COMPONENT_NAMES and isinstance(w, (int, float)) and col in pesos:
                    pesos[col] = float(min(self.KERNEL_WEIGHT_MAX, max(self.KERNEL_WEIGHT_MIN, w)))
        return pesos

    def elicit_kernel_weights(self, seed: Optional[int] = None):
        """Single elicitation, before the campaign starts. Used by
        bo_kernel_shaped."""
        system_prompt = self._build_kernel_shaper_prompt()
        user_prompt = "Assign your initial importance weights for this campaign."
        respuesta = self._call_llm(system_prompt, user_prompt)
        print(f"    [debug] items propuestos: {len(respuesta.get('importance', []))}, "
              f"reasoning: {respuesta.get('reasoning', '')[:200]}")
        pesos = self.parse_kernel_weights(respuesta)
        return pesos, respuesta.get('reasoning', '')

    def elicit_kernel_weights_ard_feedback(self, X_bo, Y_bo, pesos_previos: dict, reasoning_previo: str,
                                            seed: Optional[int] = None):
        """Re-elicitation used by bo_kernel_shaped_ard_loop. Reuses
        extract_ard_lengthscales/format_ard_importance, but here they are
        called on a probe GP fit to the REAL (un-rescaled) X_bo: fitting on
        the already-rescaled X_bo would echo the LLM's own prior back to it,
        not give an independent reading of the data."""
        gp_sondeo = SingleTaskGP(X_bo, Y_bo)
        mll_sondeo = ExactMarginalLogLikelihood(gp_sondeo.likelihood, gp_sondeo)
        fit_gpytorch_mll(mll_sondeo)
        resumen_ard = self.format_ard_importance(gp_sondeo, top_k=10)

        resumen_previo = ", ".join(
            f"{name}={w:.2f}" for name, w in pesos_previos.items() if abs(w - 1.0) > 1e-6
        ) or "(no dimensions weighted away from 1.0 last time)"

        system_prompt = self._build_kernel_shaper_prompt()
        user_prompt = (
            f"This is a revision, not your first guess. Your previous weights were:\n"
            f"{resumen_previo}\nYour previous reasoning was: {reasoning_previo}\n\n"
            f"Since then, the Gaussian Process has been fit on the real experiment "
            f"history and learned its own sense of which dimensions matter most -- "
            f"the dimensions below have the SHORTEST lengthscales, meaning the data "
            f"itself found them most informative (top 10, shortest first):\n{resumen_ard}\n\n"
            f"Revise your importance weights in light of this evidence. You may keep, "
            f"strengthen, weaken, or drop any previous weight, and add new ones."
        )
        respuesta = self._call_llm(system_prompt, user_prompt)
        print(f"    [debug] items propuestos: {len(respuesta.get('importance', []))}, "
              f"reasoning: {respuesta.get('reasoning', '')[:200]}")
        pesos = self.parse_kernel_weights(respuesta)
        return pesos, respuesta.get('reasoning', '')

    def propose_candidate_kernel_shaped(self, X_bo, Y_bo, pesos_dict: dict, bounds=None,
                                         seed: Optional[int] = None):
        """Same as propose_candidate, but the GP is fit on features rescaled
        by the LLM-elicited weights, and the resulting candidate is rescaled
        back before being returned."""
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)

        pesos_tensor = torch.tensor([pesos_dict[col] for col in self.feature_columns], dtype=torch.float64)
        # Normalized by its own maximum: preserves the relative ratios
        # between weights (what actually shapes the ARD) but guarantees no
        # rescaled dimension exceeds 1.0 -- this is what avoids the BoTorch
        # warning about data outside the unit cube.
        pesos_norm = pesos_tensor / pesos_tensor.max()

        X_shaped = X_bo * pesos_norm
        bounds_shaped = bounds * pesos_norm

        gp = SingleTaskGP(X_shaped, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_fn = LogExpectedImprovement(model=gp, best_f=Y_bo.max())
        candidato_shaped, _ = optimize_acqf(
            acq_function=acq_fn, bounds=bounds_shaped, q=1, num_restarts=10, raw_samples=100
        )
        candidato = candidato_shaped / pesos_norm
        candidato = torch.clamp(candidato, min=0.0, max=1.0)  # defensive
        return candidato

    def run_kernel_shaped_iteration(self, X_bo, Y_bo, pesos_dict: dict, bounds=None,
                                     seed: Optional[int] = None) -> dict:
        if bounds is None:
            bounds = self.bounds
        candidato_tensor = self.propose_candidate_kernel_shaped(X_bo, Y_bo, pesos_dict, bounds, seed=seed)
        y_final = self.objective(candidato_tensor)
        return {'x': candidato_tensor, 'y': y_final, 'fuente': 'bo_kernel_shaped',
                'rechazos': 0, 'reasoning': '', 'log_rechazos': []}

    def run_kernel_shaping_iteration(self, metodo: str, X_bo, Y_bo, iteracion: int, seed_base: int,
                                      weights_state: dict) -> dict:
        """weights_state is a mutable dict carried between iterations of ONE
        campaign (one method, one seed): weights, reasoning, and which
        iteration last elicited them."""
        iter_seed = seed_base * 1000 + iteracion

        if metodo == 'bo_kernel_shaped':
            if weights_state['pesos'] is None:
                weights_state['pesos'], weights_state['reasoning'] = self.elicit_kernel_weights(seed=seed_base)
                no_neutros = {k: round(v, 2) for k, v in weights_state['pesos'].items() if abs(v - 1.0) > 1e-6}
                print(f"  kernel_shaped -- pesos iniciales: {no_neutros}")
            return self.run_kernel_shaped_iteration(X_bo, Y_bo, weights_state['pesos'], seed=iter_seed)

        elif metodo == 'bo_kernel_shaped_ard_loop':
            necesita_elicitar = (
                weights_state['pesos'] is None
                or (iteracion - weights_state['ultima_elicitacion']) >= self.KERNEL_ARD_REELICIT_EVERY
            )
            if necesita_elicitar:
                if weights_state['pesos'] is None:
                    weights_state['pesos'], weights_state['reasoning'] = self.elicit_kernel_weights(seed=seed_base)
                else:
                    weights_state['pesos'], weights_state['reasoning'] = self.elicit_kernel_weights_ard_feedback(
                        X_bo, Y_bo, weights_state['pesos'], weights_state['reasoning'], seed=iter_seed
                    )
                weights_state['ultima_elicitacion'] = iteracion
                no_neutros = {k: round(v, 2) for k, v in weights_state['pesos'].items() if abs(v - 1.0) > 1e-6}
                print(f"  kernel_shaped_ard_loop -- iter={iteracion} pesos: {no_neutros}")
            resultado = self.run_kernel_shaped_iteration(X_bo, Y_bo, weights_state['pesos'], seed=iter_seed)
            resultado['fuente'] = 'bo_kernel_shaped_ard_loop'
            return resultado

        else:
            raise ValueError(f"Unknown kernel-shaping method: {metodo}")
