"""Shared pytest fixtures for pipeline smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas import GenerationRecord, PrepItem, ValidationRecord


@pytest.fixture
def sample_prep_item() -> PrepItem:
    return PrepItem(
        item_id="test_item_01",
        genre="Europarl",
        arg1="The first argument sentence with surrounding context.",
        arg2="The second argument sentence following the first.",
        arg1_singlesentence="The first argument sentence.",
        arg2_singlesentence="The second argument sentence.",
        annotator_step2_senses=["cause", "cause", "cause", "conjunction"],
        n_valid_annotations=4,
        crowd_sense_distribution={"cause": 0.75, "conjunction": 0.25},
        candidate_senses=["cause", "conjunction"],
        majority_single_sense="cause",
        wikipedia_reference_labels=None,
        crowd_agreement_score=0.75,
        stratification_bin="high",
        split="train",
    )


@pytest.fixture
def filled_definitions(tmp_path: Path) -> Path:
    """Sense definitions file with two senses marked filled=true."""
    data = {
        "_meta": {"filled": True},
        "cause": {
            "definition": "ARG1 gives the cause for ARG2 or vice versa.",
            "example": "The road is icy. I drove slowly.",
            "filled": True,
        },
        "conjunction": {
            "definition": "ARG1 and ARG2 simply hold together.",
            "example": "She painted the kitchen. He fixed the sink.",
            "filled": True,
        },
    }
    p = tmp_path / "sense_definitions.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def empty_definitions(tmp_path: Path) -> Path:
    data = {
        "_meta": {"filled": False},
        "cause": {"definition": "", "example": "", "filled": False},
    }
    p = tmp_path / "sense_definitions_empty.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def sample_generation(sample_prep_item) -> GenerationRecord:
    return GenerationRecord(
        generation_id="gen-1",
        item_id=sample_prep_item.item_id,
        candidate_sense="cause",
        generator_model="openai:gpt-4o-mini",
        explanations=[
            "Arg1 describes a state that produces the action in Arg2.",
            "There is a causal chain from Arg1 to Arg2.",
        ],
        abstained=False,
        abstention_reason=None,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=400,
        timestamp="2026-05-22T10:00:00Z",
        raw_response="...",
    )


@pytest.fixture
def sample_validations(sample_generation) -> list[ValidationRecord]:
    return [
        ValidationRecord(
            validation_id=f"val-{i}",
            generation_id=sample_generation.generation_id,
            item_id=sample_generation.item_id,
            candidate_sense="cause",
            validator_model="openai:gpt-4o-mini",
            explanation_text=expl,
            validity_score=score,
            raw_response=str(score),
            parsing_success=True,
            input_tokens=80,
            output_tokens=2,
            cost_usd=0.0005,
            latency_ms=200,
            timestamp="2026-05-22T10:01:00Z",
        )
        for i, (expl, score) in enumerate(
            [
                (sample_generation.explanations[0], 0.85),
                (sample_generation.explanations[1], 0.70),
            ]
        )
    ]
