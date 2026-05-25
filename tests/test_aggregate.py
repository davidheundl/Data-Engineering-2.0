"""Smoke tests for Stage 4 (Aggregate) — KLD and distribution building."""
from __future__ import annotations

import math

from src.aggregate import _build_llm_distribution, _kld
from src.schemas import SenseStats


def test_kld_zero_for_identical_distributions():
    p = {"a": 0.5, "b": 0.5}
    assert _kld(p, p) == 0.0


def test_kld_positive_for_different_distributions():
    p = {"a": 1.0}
    q = {"b": 1.0}
    assert _kld(p, q) > 0


def _stats(mv: float) -> SenseStats:
    return SenseStats(
        max_validity=mv,
        mean_validity=mv,
        n_explanations=1,
        n_generators_abstained=0,
        validator_std=0.0,
        commission_score=1.0 - mv,
    )


def test_llm_distribution_equal_mass_normalized():
    stats = {
        "cause": _stats(0.9),
        "conjunction": _stats(0.7),
        "contrast": _stats(0.2),
    }
    dist = _build_llm_distribution(stats, tau=0.5)
    assert set(dist.keys()) == {"cause", "conjunction"}
    assert math.isclose(sum(dist.values()), 1.0)
    assert dist["cause"] == dist["conjunction"]


def test_llm_distribution_empty_when_none_validated():
    stats = {"cause": _stats(0.1)}
    dist = _build_llm_distribution(stats, tau=0.5)
    assert dist == {}
