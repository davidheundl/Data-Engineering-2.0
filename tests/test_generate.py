"""Smoke tests for Stage 2 (Generate) — JSON parsing."""
from __future__ import annotations

from src.generate import _parse_generation_response


def test_parses_clean_json():
    text = '{"explanations": ["one", "two"], "abstained": false, "abstention_reason": null}'
    expl, abs_, reason = _parse_generation_response(text)
    assert expl == ["one", "two"]
    assert abs_ is False
    assert reason is None


def test_parses_markdown_fenced_json():
    text = '```json\n{"explanations": ["x"], "abstained": false, "abstention_reason": null}\n```'
    expl, abs_, _ = _parse_generation_response(text)
    assert expl == ["x"]
    assert abs_ is False


def test_empty_explanations_treated_as_abstention():
    # Even with the no-abstention prompt, the model may still return an empty
    # list — we mark it abstained so Stage 4 can see it.
    text = '{"explanations": []}'
    _, abs_, _ = _parse_generation_response(text)
    assert abs_ is True


def test_malformed_response_kept_as_explanation():
    # New behaviour: if the model returns prose instead of JSON, treat the
    # whole text as a single explanation rather than dropping it.
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
