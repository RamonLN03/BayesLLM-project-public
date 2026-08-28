"""Local LLM client utilities shared by every agent architecture.

Both functions here are direct, byte-for-byte extractions of code that was
duplicated identically across ``dia10.ipynb``, ``dia11_prompt_sensitivity.ipynb``,
``dia12_ucb_vs_ei.ipynb``, and ``dia13_aylation_benchmark.ipynb`` (confirmed by
programmatic diff before extraction) -- only ``call_llm``'s default ``model``
value differed between notebooks (``qwen3:30b`` vs ``qwen2.5:7b``), which is
why callers pass their model explicitly rather than relying on the default.
"""

from __future__ import annotations

import json
import re

import ollama


def parse_llm_json(raw: str) -> dict:
    """Clean the LLM output and convert it to a dict.

    Tolerant of code fences and any leading/trailing prose around the JSON
    object -- finds the first ``{...}`` block via regex rather than assuming
    the response starts exactly with a code fence. Also strips a leading
    ``<think>...</think>`` reasoning trace if the model emitted one despite
    ``/no_think``.
    """
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("No JSON object found in response", raw, 0)
    return json.loads(match.group(0))


def call_llm(system_prompt: str, user_prompt: str, model: str) -> dict:
    """Call the local Ollama model and return the parsed JSON response.

    Appends ``/no_think`` to the system prompt (found to be more reliable
    than the API's ``think=False`` parameter alone at suppressing Qwen3's
    hidden reasoning trace) and passes ``think=False`` explicitly, falling
    back to a call without it for Ollama client versions that predate that
    parameter. Retries up to 3 times on a JSON-parse failure.
    """
    system_prompt = system_prompt + "\n\n/no_think"
    last_raw = None
    for attempt in range(3):
        try:
            respuesta = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                think=False,
                keep_alive=-1,
            )
        except TypeError:
            respuesta = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                keep_alive=-1,
            )
        last_raw = respuesta["message"]["content"]
        try:
            return parse_llm_json(last_raw)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"LLM did not return valid JSON after 3 attempts. Last raw output:\n{last_raw}")
