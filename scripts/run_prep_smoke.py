"""End-to-end smoke test of Stage 1 (Prep) against the real DiscoGeM CSVs.

Runs only Stage 1 — no LLM calls. Verifies that:
  - The wide + full datasets load and join correctly.
  - The crowd Level-2 distributions are computed.
  - Stratified sampling fills the configured quotas.
  - The resulting items.jsonl passes PrepItem schema validation.

Usage:
    python scripts/run_prep_smoke.py [config_path]

Default config: configs/full.yaml (50 items).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pipeline import prepare_run_dir
from src.prep import read_items, run_prep


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "configs" / "full.yaml"
    config = load_config(config_path)
    run_dir = prepare_run_dir(config, config_path, project_root)
    items_path = run_prep(config, run_dir, project_root)

    items = read_items(items_path)
    print(f"\nLoaded {len(items)} items back via PrepItem schema validation.")
    # Show first item's structure (truncated)
    first = items[0]
    print(f"\nExample item: {first.item_id} ({first.genre}, {first.stratification_bin})")
    print(f"  candidate_senses: {first.candidate_senses}")
    print(f"  crowd_distribution: {first.crowd_sense_distribution}")
    print(f"  agreement: {first.crowd_agreement_score:.3f}")
    print(f"  arg1[:120]: {first.arg1[:120]}...")


if __name__ == "__main__":
    main()
