"""Prompt templates for Stages 2 (Generate) and 3 (Validate).

Prompts are adapted from EVADE (Zuo, Plank, Peng 2025) Figures 3 and 4, with
inline PDTB 3.0 Level-2 sense definitions and canonical examples loaded from
prompts/pdtb_sense_definitions.json.

Fail-loud principle: if a candidate sense is not present in the definitions
file or has filled=false, build_generation_prompt / build_validation_prompt
raises SenseDefinitionMissing — the pipeline must not proceed with empty
definitions.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


class SenseDefinitionMissing(ValueError):
    """Raised when a candidate sense has no usable definition."""


@lru_cache(maxsize=4)
def load_sense_definitions(path: str) -> dict[str, dict]:
    """Load and cache the sense definitions JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sense definitions file not found: {p}")
    with p.open("r") as f:
        return json.load(f)


def _lookup_sense(definitions: dict[str, dict], sense: str) -> dict:
    if sense not in definitions:
        raise SenseDefinitionMissing(
            f"Sense '{sense}' is not present in the definitions file. "
            f"Available senses: {sorted(k for k in definitions if not k.startswith('_'))}"
        )
    entry = definitions[sense]
    if not entry.get("filled", False):
        raise SenseDefinitionMissing(
            f"Sense '{sense}' has filled=false. Please provide the verbatim PDTB 3.0 "
            f"definition and a canonical example before running Stage 2 or 3."
        )
    if not entry.get("definition") or not entry.get("example"):
        raise SenseDefinitionMissing(
            f"Sense '{sense}' has empty definition or example. Cannot proceed."
        )
    return entry


# ---------------------------------------------------------------------------
# Stage 2: Generation prompts
# ---------------------------------------------------------------------------
GENERATION_SYSTEM = (
    "You are an expert in discourse relation analysis under the "
    "Penn Discourse Treebank 3.0 framework."
)


def build_generation_prompt(
    sense: str, arg1: str, arg2: str, definitions_path: str
) -> tuple[str, str]:
    """Return (system_message, user_message) for explanation generation."""
    definitions = load_sense_definitions(definitions_path)
    entry = _lookup_sense(definitions, sense)

    user = (
        f"The candidate discourse relation sense is {sense}.\n\n"
        f"Definition: {entry['definition']}\n\n"
        f"Canonical example: {entry['example']}\n\n"
        f"Provide the single best explanation for why the implicit discourse "
        f"relation between Argument 1 and Argument 2 below could express "
        f"{sense}. Even if the relation seems weak or unlikely for this pair, "
        f"produce the best possible justification a careful annotator could "
        f"give for {sense}. Do not refuse or say the relation is "
        f"unjustifiable — always argue the case as well as you can.\n\n"
        f"Do not include introductory phrases. Output strictly valid JSON.\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        f"Output JSON schema:\n"
        f'{{\n'
        f'  "explanation": "..."\n'
        f'}}'
    )
    return GENERATION_SYSTEM, user


# ---------------------------------------------------------------------------
# Stage 3: Validation prompts
# ---------------------------------------------------------------------------
VALIDATION_SYSTEM = (
    "You are an expert linguistic annotator under the PDTB 3.0 framework."
)


def build_validation_prompt(
    sense: str,
    arg1: str,
    arg2: str,
    explanation: str,
    definitions_path: str,
) -> tuple[str, str]:
    """Return (system_message, user_message) for explanation validation."""
    definitions = load_sense_definitions(definitions_path)
    entry = _lookup_sense(definitions, sense)

    user = (
        f"You are evaluating whether an explanation correctly justifies "
        f"assigning a specific discourse relation label to a pair of text "
        f"arguments.\n\n"
        f"Discourse relation sense: {sense}\n"
        f"Definition: {entry['definition']}\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        f"Explanation for why this pair expresses '{sense}': {explanation}\n\n"
        f"Rate the explanation on a 0\u201310 integer scale:\n"
        f"  0\u20132: The explanation is poor \u2014 it does not convincingly justify "
        f"'{sense}' for this argument pair, or the relation clearly does not "
        f"hold.\n"
        f"  3\u20135: The explanation is mediocre \u2014 there is a weak or partial "
        f"connection, but it is not compelling.\n"
        f"  6\u20137: The explanation is reasonable \u2014 it makes a fair case for "
        f"'{sense}', though some aspects could be stronger.\n"
        f"  8\u201310: The explanation is strong \u2014 it clearly and convincingly "
        f"justifies '{sense}' for this specific argument pair.\n\n"
        f"Be genuinely critical. If the discourse relation does not fit this "
        f"argument pair well, the explanation cannot be good regardless of how "
        f"well-written it is. Give ONLY a single integer (0\u201310), no other "
        f"text.\n\n"
        f"Score:"
    )
    return VALIDATION_SYSTEM, user
