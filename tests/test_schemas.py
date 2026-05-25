"""Schema validation tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import PrepItem


def test_prep_item_distribution_must_sum_to_one(sample_prep_item):
    """Distribution that doesn't sum to ~1.0 must fail validation."""
    data = sample_prep_item.model_dump()
    data["crowd_sense_distribution"] = {"cause": 0.5}  # sums to 0.5
    with pytest.raises(ValidationError):
        PrepItem(**data)


def test_prep_item_agreement_clamped():
    """Agreement score outside [0,1] must fail."""
    with pytest.raises(ValidationError):
        PrepItem(
            item_id="x",
            genre="Lit",
            arg1="a",
            arg2="b",
            arg1_singlesentence="a",
            arg2_singlesentence="b",
            annotator_step2_senses=["cause"],
            n_valid_annotations=1,
            crowd_sense_distribution={"cause": 1.0},
            candidate_senses=["cause"],
            majority_single_sense="cause",
            crowd_agreement_score=1.5,  # invalid
            stratification_bin="high",
        )
