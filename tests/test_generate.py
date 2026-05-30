"""Smoke tests for Stage 2 (Generate) — JSON parsing."""
from __future__ import annotations

from src.generate import _parse_generation_response


def test_parses_single_explanation_json():
    text = '{"explanation": "Because Arg1 causes Arg2."}'
    expl, abs_, reason = _parse_generation_response(text)
    assert expl == ["Because Arg1 causes Arg2."]
    assert abs_ is False
    assert reason is None


def test_parses_legacy_list_takes_first():
    text = '{"explanations": ["one", "two"]}'
    expl, abs_, reason = _parse_generation_response(text)
    assert expl == ["one"]
    assert abs_ is False


def test_parses_markdown_fenced_json():
    text = '```json\n{"explanation": "x"}\n```'
    expl, abs_, _ = _parse_generation_response(text)
    assert expl == ["x"]
    assert abs_ is False


def test_empty_explanation_treated_as_abstention():
    text = '{"explanation": ""}'
    _, abs_, _ = _parse_generation_response(text)
    assert abs_ is True


def test_malformed_response_kept_as_explanation():
    text = "Sorry, I cannot output JSON."
    expl, abs_, reason = _parse_generation_response(text)
    assert expl == ["Sorry, I cannot output JSON."]
    assert abs_ is False
    assert reason is None


def test_completely_empty_response_is_abstention():
    expl, abs_, reason = _parse_generation_response("")
    assert expl == []
    assert abs_ is True
    assert reason is not None
