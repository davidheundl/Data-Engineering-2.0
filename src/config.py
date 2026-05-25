"""Configuration loading and validation.

Configs are YAML files in configs/. Three tiers are supported:
  - pilot.yaml:   10 items, 1 generator, 1 validator
  - midterm.yaml: 20 items, 2 generators, 2 validators
  - full.yaml:    50 items, 4 generators, 4 validators

API keys are resolved from environment variables — never stored in config files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    generators: list[str]
    validators: list[str]


class SamplingConfig(BaseModel):
    n_items: int = Field(ge=1)
    genre_split: dict[Literal["Europarl", "Lit", "Wiki"], int]
    seed: int = 42

    def total_items(self) -> int:
        return sum(self.genre_split.values())


class PipelineConfig(BaseModel):
    tau_start: float = 0.1
    tau_stop: float = 0.9
    tau_step: float = 0.1
    max_retries: int = 5
    retry_delay: float = 2.0
    concurrency_per_provider: int = 4
    temperature_generate: float = 0.3
    max_tokens_generate: int = 800
    max_tokens_validate: int = 10

    def tau_values(self) -> list[float]:
        """Return tau values for the threshold sweep, rounded to 2 decimals."""
        values: list[float] = []
        tau = self.tau_start
        while tau <= self.tau_stop + 1e-9:
            values.append(round(tau, 2))
            tau += self.tau_step
        return values


class DataConfig(BaseModel):
    wide_csv: str
    full_csv: str
    sense_definitions: str
    results_dir: str = "results"
    logs_dir: str = "logs"


class Config(BaseModel):
    name: str
    models: ModelsConfig
    sampling: SamplingConfig
    pipeline: PipelineConfig
    data: DataConfig


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def resolve_api_keys() -> dict[str, str]:
    """Resolve API keys from environment variables.

    Returns a dict mapping provider name to key. Missing keys are returned as
    empty strings — downstream code should check before making calls.
    """
    return {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "mistral": os.environ.get("MISTRAL_API_KEY", ""),
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
    }
