"""Pydantic schemas for all pipeline stage outputs.

All JSONL records are validated against these schemas on read and write.
The pipeline fails loudly on schema violations.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# PDTB 3.0 Level-2 sense vocabulary
# Derived from observed values in DiscoGeMcorpus_fulldataset.csv (lev2_conn2).
# ---------------------------------------------------------------------------
LEVEL2_SENSES: list[str] = [
    "cause",
    "cause+belief",
    "cause+speechact",
    "condition",
    "condition+speechact",
    "negative-condition",
    "purpose",
    "concession",
    "concession+speechact",
    "contrast",
    "similarity",
    "synchronous",
    "asynchronous",
    "conjunction",
    "disjunction",
    "equivalence",
    "exception",
    "instantiation",
    "level-of-detail",
    "manner",
    "substitution",
    "norel",
]

Genre = Literal["Europarl", "Lit", "Wiki"]
AgreementBin = Literal["high", "low"]


# ---------------------------------------------------------------------------
# Stage 1: Prep
# ---------------------------------------------------------------------------
class PrepItem(BaseModel):
    """One row of items.jsonl."""

    item_id: str
    genre: Genre
    arg1: str
    arg2: str
    arg1_singlesentence: str
    arg2_singlesentence: str
    annotator_step2_senses: list[str]
    n_valid_annotations: int = Field(ge=1)
    crowd_sense_distribution: dict[str, float]
    candidate_senses: list[str]
    majority_single_sense: str
    wikipedia_reference_labels: list[str] | None = None
    crowd_agreement_score: float = Field(ge=0.0, le=1.0)
    stratification_bin: AgreementBin
    split: str | None = None

    @field_validator("crowd_sense_distribution")
    @classmethod
    def _distribution_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"crowd_sense_distribution must sum to 1.0, got {total}")
        return v


# ---------------------------------------------------------------------------
# Stage 2: Generate
# ---------------------------------------------------------------------------
class GenerationRecord(BaseModel):
    """One row of generations.jsonl."""

    generation_id: str
    item_id: str
    candidate_sense: str
    generator_model: str  # e.g. "openai:gpt-4o-mini"
    explanations: list[str]
    abstained: bool
    abstention_reason: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    timestamp: str
    raw_response: str


# ---------------------------------------------------------------------------
# Stage 3: Validate
# ---------------------------------------------------------------------------
class ValidationRecord(BaseModel):
    """One row of validations.jsonl."""

    validation_id: str
    generation_id: str
    item_id: str
    candidate_sense: str
    validator_model: str
    explanation_text: str
    validity_score: float = Field(ge=0.0, le=1.0)
    raw_response: str
    parsing_success: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    timestamp: str


# ---------------------------------------------------------------------------
# Stage 4: Aggregate
# ---------------------------------------------------------------------------
class SenseStats(BaseModel):
    max_validity: float = Field(ge=0.0, le=1.0)
    mean_validity: float = Field(ge=0.0, le=1.0)
    n_explanations: int = Field(ge=0)
    n_generators_abstained: int = Field(ge=0)
    validator_std: float = Field(ge=0.0)
    commission_score: float = Field(ge=0.0, le=1.0)


class DistributionRecord(BaseModel):
    """One row of distributions.jsonl."""

    item_id: str
    candidate_senses: list[str]
    per_sense_stats: dict[str, SenseStats]
    llm_label_distribution_per_tau: dict[str, dict[str, float]]


class RunMetrics(BaseModel):
    """metrics.json — global metrics across a run."""

    run_id: str
    config_name: str
    n_items: int
    total_generation_calls: int
    total_validation_calls: int
    total_cost_usd: float
    mean_cost_per_item: float
    abstention_rate_per_model: dict[str, float]
    avg_commission_score_by_genre: dict[str, float]
    kld_vs_crowd_per_tau: dict[str, float]
    tau_values: list[float]


# ---------------------------------------------------------------------------
# Cross-cutting: LLM call log
# ---------------------------------------------------------------------------
class LLMCallLog(BaseModel):
    """One row of logs/llm_calls_{run_id}.jsonl."""

    stage: str
    provider: str
    model: str
    item_id: str | None = None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    timestamp: str
    success: bool
    error: str | None = None
