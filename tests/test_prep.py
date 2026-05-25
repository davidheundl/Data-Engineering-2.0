"""Smoke tests for Stage 1 (Prep)."""
from __future__ import annotations

import math

from src.prep import _build_crowd_distribution, _split_high_low


def test_crowd_distribution_normalizes_to_one():
    dist = _build_crowd_distribution(["cause", "cause", "conjunction", "result"])
    assert math.isclose(sum(dist.values()), 1.0)
    assert dist["cause"] == 0.5
    assert dist["conjunction"] == 0.25
    assert dist["result"] == 0.25


def test_split_high_low_balance():
    assert _split_high_low(16) == (8, 8)
    assert _split_high_low(17) == (9, 8)  # extra goes to high
