"""Pipeline orchestrator: run all 5 stages sequentially.

A run_id is derived from a UTC timestamp + short git commit hash (best-effort).
All outputs go to results/{run_id}/. The config used and the sense definitions
file are copied into the run directory for reproducibility.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .aggregate import run_aggregate
from .analyze import run_analyze
from .config import Config, load_config
from .generate import run_generate
from .prep import run_prep
from .validate import run_validate


def _git_hash(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "nogit"


def make_run_id(config_name: str, project_root: Path) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{config_name}_{_git_hash(project_root)}"


def prepare_run_dir(config: Config, config_path: Path, project_root: Path) -> Path:
    """Create results/{run_id}/, copy config + sense definitions, return path."""
    run_id = make_run_id(config.name, project_root)
    run_dir = project_root / config.data.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Copy sense definitions
    senses_src = project_root / config.data.sense_definitions
    if senses_src.exists():
        shutil.copy2(senses_src, run_dir / Path(senses_src).name)

    return run_dir


def find_run_dir(config: Config, project_root: Path, run_id: str) -> Path:
    p = project_root / config.data.results_dir / run_id
    if not p.exists():
        raise FileNotFoundError(f"Run dir not found: {p}")
    return p


async def run_pipeline(
    config_path: Path,
    project_root: Path,
    *,
    stages: list[str],
    run_id: str | None = None,
) -> Path:
    """Run a subset of stages. If run_id is given, resume into that run dir."""
    config = load_config(config_path)
    if run_id is None:
        run_dir = prepare_run_dir(config, config_path, project_root)
    else:
        run_dir = find_run_dir(config, project_root, run_id)

    print(f"=== Run: {run_dir.name} (config: {config.name}) ===")

    if "prep" in stages or "1" in stages:
        run_prep(config, run_dir, project_root)
    if "generate" in stages or "2" in stages:
        await run_generate(config, run_dir, project_root)
    if "validate" in stages or "3" in stages:
        await run_validate(config, run_dir, project_root)
    if "aggregate" in stages or "4" in stages:
        run_aggregate(config, run_dir, project_root)
    if "analyze" in stages or "5" in stages:
        run_analyze(config, run_dir, project_root)

    return run_dir
