"""Smoke tests for Stage 3 (Validate) — score parsing."""
from __future__ import annotations

from src.validate import parse_validity_score


def test_parses_plain_float():
    score, ok = parse_validity_score("0.87")
    assert ok is True
    assert score == 0.87


def test_parses_with_whitespace():
    score, ok = parse_validity_score("  0.5  ")
    assert ok is True
    assert score == 0.5


def test_normalizes_zero_to_ten_scale():
    score, ok = parse_validity_score("7")
    assert ok is True
    assert score == 0.7


def test_clamps_to_one():
    score, ok = parse_validity_score("1.2")
    assert ok is True
    assert score == 1.0


def test_clamps_to_zero():
    score, ok = parse_validity_score("-0.1")
    assert ok is True
    assert score == 0.0


def test_parses_with_extra_words():
    """Some models stubbornly add text; we extract the first float."""
    score, ok = parse_validity_score("Probability: 0.42 (because ...)")
    assert ok is True
    assert score == 0.42


def test_parsing_failure_returns_zero():
    score, ok = parse_validity_score("no number here")
    assert ok is False
    assert score == 0.0
