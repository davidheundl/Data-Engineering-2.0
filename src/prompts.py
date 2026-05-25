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
        f"List every distinct explanation for why the implicit discourse relation "
        f"between Argument 1 and Argument 2 below could express {sense}. "
        f"You MUST provide at least one explanation. Even if the relation seems "
        f"weak or unlikely for this pair, produce the best possible justification "
        f"a careful annotator could give for {sense}. Do not refuse, do not "
        f"return an empty list, do not say the relation is unjustifiable — always "
        f"argue the case as well as you can.\n\n"
        f"Do not paraphrase the same idea in different words. Do not include "
        f"introductory phrases. Output strictly valid JSON.\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        f"Output JSON schema:\n"
        f'{{\n'
        f'  "explanations": ["...", "..."]\n'
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
        f"We have collected an annotation for a discourse relation together "
        f"with a reason for the label. Your task is to judge whether the reason "
        f"makes sense for the label. Provide the probability (0.0 to 1.0) that "
        f"the reason makes sense for the label. Give ONLY the probability as a "
        f"number, no other words or explanation.\n\n"
        f"Sense: {sense}\n"
        f"Definition: {entry['definition']}\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        f"Reason for label {sense}: {explanation}\n\n"
        f"Probability:"
    )
    return VALIDATION_SYSTEM, user
