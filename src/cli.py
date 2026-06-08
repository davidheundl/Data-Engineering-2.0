"""Command-line interface for the EVADE-on-DiscoGeM pipeline.

Examples:
    python -m src.cli run --config configs/pilot.yaml
    python -m src.cli run --config configs/full.yaml --stages 2,3,4,5 --run-id 20260521T...
    python -m src.cli analyze --config configs/pilot.yaml --run-id 20260521T...
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .pipeline import run_pipeline


def _project_root() -> Path:
    """Project root = parent of the src/ directory."""
    return Path(__file__).resolve().parent.parent


def _parse_stages(arg: str | None) -> list[str]:
    if arg is None or arg.lower() == "all":
        return ["prep", "generate", "validate", "aggregate", "analyze"]
    return [s.strip() for s in arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="EVADE-on-DiscoGeM pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run pipeline stages")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.add_argument(
        "--stages",
        default="all",
        help='Comma-separated stages: prep,generate,validate,aggregate,analyze (or "all")',
    )
    p_run.add_argument("--run-id", default=None, help="Resume into existing run dir")
    p_run.add_argument("--fork-from", default=None, help="Copy data from existing run into new dir, then run stages")
    p_run.add_argument("--fork-name", default=None, help="Custom name for the forked run directory")

    p_an = sub.add_parser("analyze", help="Run analyze stage only")
    p_an.add_argument("--config", required=True)
    p_an.add_argument("--run-id", required=True)

    args = parser.parse_args()
    project_root = _project_root()

    if args.command == "run":
        stages = _parse_stages(args.stages)
        asyncio.run(
            run_pipeline(
                Path(args.config),
                project_root,
                stages=stages,
                run_id=args.run_id,
                fork_from=args.fork_from,
                fork_name=args.fork_name,
            )
        )
    elif args.command == "analyze":
        asyncio.run(
            run_pipeline(
                Path(args.config),
                project_root,
                stages=["analyze"],
                run_id=args.run_id,
            )
        )


if __name__ == "__main__":
    main()
