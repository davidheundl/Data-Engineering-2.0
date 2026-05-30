"""Smoke tests for Stage 3 (Validate) — 0–10 integer score parsing."""
from __future__ import annotations

from src.validate import parse_validity_score


def test_parses_integer_score():
    score, ok = parse_validity_score("7")
    assert ok is True
    assert score == 0.7


def test_parses_with_whitespace():
    score, ok = parse_validity_score("  8  ")
    assert ok is True
    assert score == 0.8


def test_parses_zero():
    score, ok = parse_validity_score("0")
    assert ok is True
    assert score == 0.0


def test_parses_ten():
    score, ok = parse_validity_score("10")
    assert ok is True
    assert score == 1.0


def test_clamps_above_ten():
    score, ok = parse_validity_score("12")
    assert ok is True
    assert score == 1.0


def test_parses_with_extra_text():
    """Some models stubbornly add text; we extract the first integer."""
    score, ok = parse_validity_score("Score: 5 because ...")
    assert ok is True
    assert score == 0.5


def test_parsing_failure_returns_zero():
    score, ok = parse_validity_score("no number here")
    assert ok is False
    assert score == 0.0
