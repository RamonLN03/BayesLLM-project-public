"""bayesllm -- shared library for the LLM-agent / Bayesian-optimisation
architectures benchmarked in the paper.

This package factors out the code that was byte-for-byte duplicated across
several of the original research notebooks (data loading, GP-BO candidate
proposal, the Proposer-Critic-Verifier deliberation loop, etc.) into
reusable modules and classes, so each experiment notebook only needs to
contain what is actually unique to that experiment: its configuration
(seeds, iteration counts, model name) and, where applicable, a small amount
of experiment-specific orchestration code.

Every function/method here is a direct, behaviour-preserving extraction
from the original notebooks -- see ``experiments/README.md`` for the
notebook-to-module mapping and the verification performed.
"""
