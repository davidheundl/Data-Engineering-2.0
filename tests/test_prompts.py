"""Tests for prompt building and the fail-loud definitions check."""
from __future__ import annotations

import pytest

from src.prompts import SenseDefinitionMissing, build_generation_prompt, build_validation_prompt, load_sense_definitions


def test_generation_prompt_uses_full_context(filled_definitions, sample_prep_item):
    sys, user = build_generation_prompt(
        "cause",
        sample_prep_item.arg1,
        sample_prep_item.arg2,
        str(filled_definitions),
    )
    assert sample_prep_item.arg1 in user
    assert sample_prep_item.arg2 in user
    assert "cause" in user
    assert "JSON" in user


def test_validation_prompt_is_probability_only(filled_definitions, sample_prep_item):
    _, user = build_validation_prompt(
        "cause",
        sample_prep_item.arg1,
        sample_prep_item.arg2,
        "Because Arg1 caused Arg2.",
        str(filled_definitions),
    )
    assert "Probability:" in user
    assert "ONLY the probability" in user


def test_fail_loud_on_unfilled_definition(empty_definitions, sample_prep_item):
    # Bust the lru_cache by using a different path
    load_sense_definitions.cache_clear()
    with pytest.raises(SenseDefinitionMissing):
        build_generation_prompt("cause", "a", "b", str(empty_definitions))


def test_fail_loud_on_unknown_sense(filled_definitions):
    load_sense_definitions.cache_clear()
    with pytest.raises(SenseDefinitionMissing):
        build_generation_prompt("not-a-real-sense", "a", "b", str(filled_definitions))
