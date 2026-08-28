"""Direct (Pd-catalysed) C-H arylation benchmark, and the LLM agent
architectures benchmarked against it (PCV, BatchSelect, each with an
EI and a UCB acquisition variant).

This module is a behaviour-preserving, method-per-function extraction of
``dia13_aylation_benchmark.ipynb`` -- kept in its own class (mirroring
``BuchwaldHartwigBenchmark``'s method names for consistency) rather than
merged into it, because although the two benchmarks share the same overall
agent architecture, essentially every function differs in some real detail
(a different, HTTP-fetched dataset; a different, row-wise, list-based
feature-vector builder; a clipped-to-[0,100] objective; no SingleAgent or
BatchSelect-random/top-acquisition control arms). This benchmark was added
specifically to check whether findings on Buchwald-Hartwig generalised to a
second, mechanistically different reaction -- see the paper's
"Generalisation: Direct Arylation" section.
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

DATASET_URL = (
    "https://raw.githubusercontent.com/b-shields/edbo/refs/heads/master/"
    "experiments/data/direct_arylation/experiment_index.csv"
)

RDKIT_DESCRIPTORS = [
    'LabuteASA', 'NumRotatableBonds', 'FractionCSP3', 'NumAromaticRings',
    'RingCount', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
    'TPSA', 'HeavyAtomCount',
]


def _compute_descriptors(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return [getattr(Descriptors, name)(mol) for name in RDKIT_DESCRIPTORS]


class DirectArylationBenchmark:
    """RandomForest emulator over the Perera et al. Direct Arylation HTE
    dataset (fetched from the public `edbo` mirror; 3 categorical
    components -- ligand, base, solvent -- each described by 10 RDKit
    descriptors, plus continuous concentration and temperature, for a 32-D
    space), plus the LLM agent architectures benchmarked against it: PCV
    and BatchSelect, each with an EI and a UCB acquisition variant.

    Parameters
    ----------
    model:
        Ollama model name passed to every LLM call made through this
        instance.
    dataset_url:
        Overridable for testing; defaults to the public `edbo` mirror used
        by the original notebook.
    """

    def __init__(self, model: str = "qwen3:30b", dataset_url: str = DATASET_URL):
        self.model = model

        df = pd.read_csv(dataset_url)

        ligand_descriptors = {s: _compute_descriptors(s) for s in df['Ligand_SMILES'].unique()}
        base_descriptors = {s: _compute_descriptors(s) for s in df['Base_SMILES'].unique()}
        solvent_descriptors = {s: _compute_descriptors(s) for s in df['Solvent_SMILES'].unique()}

        self.feature_names = (
            [f"ligand_{d}" for d in RDKIT_DESCRIPTORS] +
            [f"base_{d}" for d in RDKIT_DESCRIPTORS] +
            [f"solvent_{d}" for d in RDKIT_DESCRIPTORS] +
            ["Concentration", "Temp_C"]
        )

        def build_feature_vector(row):
            v_ligand = ligand_descriptors[row['Ligand_SMILES']]
            v_base = base_descriptors[row['Base_SMILES']]
            v_solvent = solvent_descriptors[row['Solvent_SMILES']]
            continuous = [row['Concentration'], row['Temp_C']]
            return v_ligand + v_base + v_solvent + continuous  # list concatenation

        self.X_raw = np.array([build_feature_vector(row) for _, row in df.iterrows()])
        self.y_raw = df['yield'].values

        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_raw)

        self.emulator = RandomForestRegressor(n_estimators=300, random_state=42)
        self.emulator.fit(self.X_scaled, self.y_raw)

        self.n_dims = self.X_scaled.shape[1]  # 32
        self.bounds = torch.tensor([[0.0] * self.n_dims, [1.0] * self.n_dims], dtype=torch.float64)

        self.descriptor_bounds_physical = {
            name: (self.X_raw[:, i].min(), self.X_raw[:, i].max())
            for i, name in enumerate(self.feature_names)
        }

        # Disabled, same rationale as the Buchwald-Hartwig benchmark: the
        # Critic relies on the bounds check + its own semantic judgment
        # instead of this heuristic.
        self.redundancy_threshold = 0.0

    # -- LLM client -----------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        return call_llm(system_prompt, user_prompt, model=self.model)

    # -- Objective and candidate <-> dict conversions --------------------
    def objective(self, X):
        """Emulated black-box objective: X is a tensor (n, 32) of scaled
        descriptors, each dimension in [0, 1]. Returns a tensor (n, 1) of
        predicted reaction yield (%), clipped to [0, 100]."""
        X_numpy = X.detach().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_pred = self.emulator.predict(X_numpy)
        y_pred = np.clip(y_pred, 0, 100)
        return torch.tensor(y_pred, dtype=torch.float64).unsqueeze(-1)

    def dict_to_tensor(self, candidate_dict: dict) -> torch.Tensor:
        values = [candidate_dict[name] for name in self.feature_names]
        return torch.tensor([values], dtype=torch.float64)

    def tensor_to_dict(self, x_tensor: torch.Tensor) -> dict:
        x_flat = x_tensor.flatten()
        return {name: x_flat[i].item() for i, name in enumerate(self.feature_names)}

    def to_physical(self, candidate_scaled_dict: dict) -> dict:
        scaled_array = np.array([[candidate_scaled_dict[name] for name in self.feature_names]])
        physical_array = self.scaler.inverse_transform(scaled_array)[0]
        return {name: physical_array[i] for i, name in enumerate(self.feature_names)}

    def to_scaled(self, candidate_physical_dict: dict) -> dict:
        physical_array = np.array([[candidate_physical_dict[name] for name in self.feature_names]])
        scaled_array = self.scaler.transform(physical_array)[0]
        scaled_array = np.clip(scaled_array, 0.0, 1.0)
        return {name: scaled_array[i] for i, name in enumerate(self.feature_names)}

    def format_candidate(self, candidate_scaled_dict: dict) -> str:
        physical = self.to_physical(candidate_scaled_dict)
        return ", ".join(f"{name}={value:.3f}" for name, value in physical.items())

    def format_history(self, history: list) -> str:
        lines = ["Yield trend so far (iteration: yield):"]
        for i, exp in enumerate(history):
            lines.append(f"  {i + 1}: {exp['yield']:.2f}%")
        best_exp = max(history, key=lambda exp: exp['yield'])
        best_idx = history.index(best_exp) + 1
        lines.append(f"\nBest experiment so far (iteration {best_idx}, yield={best_exp['yield']:.2f}%):")
        lines.append("  " + self.format_candidate(best_exp['descriptors']))
        return "\n".join(lines)

    def format_descriptor_bounds(self) -> str:
        lines = [f"{name}: [{lo:.3f}, {hi:.3f}]" for name, (lo, hi) in self.descriptor_bounds_physical.items()]
        return "\n".join(lines)

    # -- Prompts ----------------------------------------------------------
    def _build_proposer_prompt(self) -> str:
        return f"""You are an expert chemist acting as a creative experimentalist in a
sequential Bayesian optimisation campaign for a palladium-catalysed direct C-H
arylation reaction. You receive a candidate experiment proposed by Bayesian
Optimisation, described by 32 physicochemical descriptors (in physical units)
of its three components (ligand, base, solvent), plus reaction concentration
and temperature, along with the experiment history.

You do NOT need to modify all 32 dimensions. Only propose adjustments to the
dimensions you judge chemically relevant given your reasoning (e.g. steric
descriptors like LabuteASA as a proxy for ligand bulk, or partial-charge
descriptors as a proxy for electronic donor/acceptor character). Leave
everything else unchanged; if you have no strong chemical justification for
changing a dimension, do not include it.

Any value you propose for a dimension MUST stay within the physically
observed range for that descriptor in the real dataset, shown below:
{self.format_descriptor_bounds()}

Respond ONLY with a valid JSON object:
{{
        "adjustments": [
        {{"dimension": "<one of the descriptor names shown>", "new_value": <number, physical units>}}
    ],
    "reasoning": "<string explaining your chemical rationale>"

}}
"adjustments" may be an empty list if you approve the candidate as-is.
No additional text, no markdown."""

    def _build_critic_prompt(self) -> str:
        return """You are a rigorous chemistry expert reviewing an experimental proposal
for a palladium-catalysed direct C-H arylation reaction. The proposal has
ALREADY passed automated bounds and redundancy checks (not your concern).
Your ONLY job is to evaluate whether the chemical reasoning provided is
coherent, plausible, and consistent with the experimental history shown.
Reject only if the reasoning is contradictory, nonsensical, or unsupported by
the data shown.

Respond ONLY with a valid JSON object:
{
    "approved": <true or false>,
    "rejection_reason": "<string if rejected, else null>",
    "concern_level": "<low|medium|high>"
}
No additional text, no markdown."""

    def _build_verifier_prompt(self) -> str:
        return f"""You are the lead scientist overseeing a direct C-H arylation
optimisation campaign. You receive a proposal that has passed safety and
coherence review. Make the final decision: confirm it as-is, or make a minor
adjustment to a small number of dimensions if you have strong chemical
grounds to do so. Log your reasoning for interpretability.

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

    def _build_selector_prompt(self) -> str:
        return """You are an expert chemist selecting which candidate experiment to run
next from a batch proposed by Bayesian Optimisation, for a palladium-catalysed
direct C-H arylation reaction. Each candidate is described by 32
physicochemical descriptors of its three components (ligand, base, solvent)
plus concentration and temperature. The candidates are shown in NO particular
order -- do not assume the order reflects any ranking or preference. Use your
own chemical reasoning, together with the experiment history, to judge which
candidate is most promising.

Respond ONLY with a valid JSON object:
{
    "selected_index": <integer, the "Candidate N" index of your choice>,
    "reasoning": "<string explaining your choice>"
}
No additional text, no markdown."""

    # -- Deterministic checks ---------------------------------------------
    def apply_adjustments(self, base_candidate_scaled: dict, adjustments: list):
        physical = self.to_physical(base_candidate_scaled)
        applied_dims = []
        for adj in adjustments:
            dim = adj.get('dimension')
            value = adj.get('new_value')
            if dim in self.feature_names and isinstance(value, (int, float)):
                physical[dim] = value
                applied_dims.append(dim)
        physical_array = np.array([[physical[name] for name in self.feature_names]])
        scaled_array = self.scaler.transform(physical_array)[0]
        out_of_range_dims = [self.feature_names[i] for i, v in enumerate(scaled_array) if v < 0.0 or v > 1.0]
        scaled_array_clipped = np.clip(scaled_array, 0.0, 1.0)
        candidate_scaled = {name: scaled_array_clipped[i] for i, name in enumerate(self.feature_names)}
        return candidate_scaled, out_of_range_dims, applied_dims

    def check_bounds(self, out_of_range_dims: list):
        if out_of_range_dims:
            return False, f"Adjustment(s) outside observed physical range: {', '.join(out_of_range_dims)}"
        return True, None

    def check_redundancy(self, candidate_scaled: dict, history: list, adjusted_dims: list,
                          threshold: Optional[float] = None):
        if threshold is None:
            threshold = self.redundancy_threshold
        k = len(adjusted_dims)
        if k == 0:
            return True, None
        thr = threshold * (k / self.n_dims) ** 0.5
        x_new = torch.tensor([candidate_scaled[d] for d in adjusted_dims], dtype=torch.float64)
        for exp in history:
            x_prev = torch.tensor([exp['descriptors'][d] for d in adjusted_dims], dtype=torch.float64)
            distance = torch.norm(x_new - x_prev).item()
            if distance < thr:
                return False, f"Too similar to a previous experiment on adjusted dims {adjusted_dims} (dist={distance:.3f}, threshold={thr:.3f})"
        return True, None

    # -- Agent calls --------------------------------------------------------
    def call_proposer(self, candidate_scaled: dict, history: list) -> dict:
        system_prompt = self._build_proposer_prompt()
        user_prompt = (
            f"{self.format_history(history)}\n\n"
            f"Bayesian Optimisation proposes the following candidate "
            f"(physical units):\n{self.format_candidate(candidate_scaled)}\n\n"
            f"Enrich this proposal with your chemical reasoning."
        )
        response = self._call_llm(system_prompt, user_prompt)
        adjustments = response.get('adjustments', [])
        merged_scaled, out_of_range_dims, applied_dims = self.apply_adjustments(candidate_scaled, adjustments)
        return {
            'descriptors': merged_scaled, 'reasoning': response.get('reasoning', ''),
            'out_of_range_dims': out_of_range_dims, 'adjusted_dims': applied_dims,
        }

    def call_critic(self, proposal: dict, history: list) -> dict:
        bounds_ok, bounds_reason = self.check_bounds(proposal['out_of_range_dims'])
        if not bounds_ok:
            return {'approved': False, 'rejection_reason': bounds_reason, 'concern_level': 'high', 'rejection_layer': 'deterministic'}

        redundancy_ok, redundancy_reason = self.check_redundancy(
            proposal['descriptors'], history, proposal['adjusted_dims']
        )
        if not redundancy_ok:
            return {'approved': False, 'rejection_reason': redundancy_reason, 'concern_level': 'medium', 'rejection_layer': 'deterministic'}

        system_prompt = self._build_critic_prompt()
        user_prompt = (
            f"{self.format_history(history)}\n\n"
            f"Proposed experiment (physical units):\n{self.format_candidate(proposal['descriptors'])}\n\n"
            f"Reasoning given: {proposal['reasoning']}\n"
            f"Evaluate the coherence of this reasoning."
        )
        verdict = self._call_llm(system_prompt, user_prompt)
        verdict['rejection_layer'] = 'llm' if not verdict.get('approved', False) else None
        return verdict

    def call_verifier(self, proposal: dict, history: list) -> dict:
        system_prompt = self._build_verifier_prompt()
        user_prompt = (
            f"{self.format_history(history)}\n\n"
            f"Approved proposal (physical units):\n{self.format_candidate(proposal['descriptors'])}\n\n"
            f"Original reasoning: {proposal['reasoning']}\n"
            f"Make your final decision."
        )
        response = self._call_llm(system_prompt, user_prompt)
        adjustments = response.get('adjustments', [])
        final_scaled, out_of_range_dims, applied_dims = self.apply_adjustments(proposal['descriptors'], adjustments)
        return {
            'descriptors': final_scaled, 'final_reasoning': response.get('final_reasoning', ''),
            'adjustment_made': response.get('adjustment_made', False),
            'out_of_range_dims': out_of_range_dims, 'adjusted_dims': applied_dims,
        }

    def call_selector(self, candidates_scaled_list: list, history: list, seed: Optional[int] = None):
        n = len(candidates_scaled_list)
        rng = np.random.default_rng(seed)
        shuffled_order = rng.permutation(n)
        candidates_text = "\n".join(
            f"Candidate {display_i}: {self.format_candidate(candidates_scaled_list[real_i])}"
            for display_i, real_i in enumerate(shuffled_order)
        )
        system_prompt = self._build_selector_prompt()
        user_prompt = (
            f"{self.format_history(history)}\n\n"
            f"Batch of candidates proposed by Bayesian Optimisation "
            f"(physical units, unordered):\n{candidates_text}\n\n"
            f"Select the most promising candidate."
        )
        response = self._call_llm(system_prompt, user_prompt)
        selected_display_idx = response.get('selected_index', 0)
        if not isinstance(selected_display_idx, int) or not (0 <= selected_display_idx < n):
            selected_display_idx = 0
        selected_real_idx = shuffled_order[selected_display_idx]
        return selected_real_idx, response.get('reasoning', '')

    # -- GP-BO candidate proposal -----------------------------------------
    def propose_candidate(self, X_bo, Y_bo, bounds=None, seed: Optional[int] = None):
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)
        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_fn = LogExpectedImprovement(model=gp, best_f=Y_bo.max())
        candidate, _ = optimize_acqf(acq_function=acq_fn, bounds=bounds, q=1, num_restarts=10, raw_samples=100)
        return candidate

    def propose_candidate_ucb(self, X_bo, Y_bo, bounds=None, beta: float = 2.0, seed: Optional[int] = None):
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)
        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_fn = UpperConfidenceBound(model=gp, beta=beta)
        candidate, _ = optimize_acqf(acq_function=acq_fn, bounds=bounds, q=1, num_restarts=10, raw_samples=100)
        return candidate

    def propose_batch(self, X_bo, Y_bo, bounds=None, q: int = 4, seed: Optional[int] = None):
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)
        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_fn = qLogNoisyExpectedImprovement(model=gp, X_baseline=X_bo)
        candidates, _ = optimize_acqf(acq_function=acq_fn, bounds=bounds, q=q, num_restarts=10, raw_samples=100)
        return candidates

    def propose_batch_ucb(self, X_bo, Y_bo, bounds=None, q: int = 4, beta: float = 2.0,
                           seed: Optional[int] = None):
        if bounds is None:
            bounds = self.bounds
        if seed is not None:
            torch.manual_seed(seed)
        gp = SingleTaskGP(X_bo, Y_bo)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_fn = qUpperConfidenceBound(model=gp, beta=beta)
        candidates, _ = optimize_acqf(acq_function=acq_fn, bounds=bounds, q=q, num_restarts=10, raw_samples=100)
        return candidates

    # -- Iteration-level orchestration -------------------------------------
    def run_pcv_deliberation(self, candidate_tensor, X_bo, Y_bo, history, bounds=None,
                              max_rejections: int = 3, iteration: int = 0, verbose: bool = False) -> dict:
        if bounds is None:
            bounds = self.bounds
        X_bo_temp = X_bo.clone()
        Y_bo_temp = Y_bo.clone()
        candidate_dict = self.tensor_to_dict(candidate_tensor)

        attempts = 0
        approved = False
        proposal = None
        rejection_log = []

        while not approved and attempts < max_rejections:
            proposal = self.call_proposer(candidate_dict, history)
            verdict = self.call_critic(proposal, history)

            if verdict['approved']:
                approved = True
            else:
                log_entry = {'attempt': attempts, 'layer': verdict.get('rejection_layer'), 'reason': verdict.get('rejection_reason')}
                rejection_log.append(log_entry)
                if verbose:
                    print(f"  Rejection [{log_entry['layer']}]: {log_entry['reason']}")

                x_rejected = self.dict_to_tensor(proposal['descriptors'])
                y_fantasy = Y_bo_temp.min().reshape(1, 1)
                X_bo_temp = torch.cat([X_bo_temp, x_rejected])
                Y_bo_temp = torch.cat([Y_bo_temp, y_fantasy])

                attempts += 1
                new_candidate = self.propose_candidate(X_bo_temp, Y_bo_temp, bounds, seed=iteration * 100 + attempts)
                candidate_dict = self.tensor_to_dict(new_candidate)

        if approved:
            decision = self.call_verifier(proposal, history)
            x_final = self.dict_to_tensor(decision['descriptors'])
            source = 'multiagent'
            final_reasoning = decision['final_reasoning']
        else:
            x_final = self.dict_to_tensor(candidate_dict)
            source = 'bo_fallback'
            final_reasoning = 'Fallback: agents did not reach consensus'

        y_final = self.objective(x_final)

        return {
            'x': x_final, 'y': y_final, 'source': source, 'rejections': attempts,
            'reasoning': final_reasoning, 'rejection_log': rejection_log,
        }

    def run_bo_only_iteration(self, X_bo, Y_bo, bounds=None, seed: Optional[int] = None) -> dict:
        if bounds is None:
            bounds = self.bounds
        candidate_tensor = self.propose_candidate(X_bo, Y_bo, bounds, seed=seed)
        y_final = self.objective(candidate_tensor)
        return {'x': candidate_tensor, 'y': y_final, 'source': 'bo_only', 'rejections': 0, 'reasoning': '', 'rejection_log': []}

    def run_bo_only_ucb_iteration(self, X_bo, Y_bo, bounds=None, beta: float = 2.0,
                                   seed: Optional[int] = None) -> dict:
        if bounds is None:
            bounds = self.bounds
        candidate_tensor = self.propose_candidate_ucb(X_bo, Y_bo, bounds, beta=beta, seed=seed)
        y_final = self.objective(candidate_tensor)
        return {'x': candidate_tensor, 'y': y_final, 'source': 'bo_only_ucb', 'rejections': 0, 'reasoning': '', 'rejection_log': []}

    def generate_initial_design(self, seed: int, n_init: Optional[int] = None):
        if n_init is None:
            n_init = 2 * self.n_dims
        X_init = draw_sobol_samples(bounds=self.bounds, n=n_init, q=1, seed=seed).squeeze(1)
        Y_init = self.objective(X_init)
        history_init = [
            {"descriptors": self.tensor_to_dict(X_init[i]), "yield": Y_init[i].item()}
            for i in range(X_init.shape[0])
        ]
        return X_init, Y_init, history_init

    @staticmethod
    def clone_run_state(X_init, Y_init, history_init):
        return X_init.clone(), Y_init.clone(), list(history_init)
