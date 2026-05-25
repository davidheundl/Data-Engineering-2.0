"""Stage 3: Validate — every validator model scores every explanation.

For each (explanation, validator_model) pair, make one LLM call. Self-
validation is allowed (the same model scoring its own explanations) — this
matches EVADE's one-expl regime.

Defensive parsing: strip whitespace, find first float, clamp to [0.0, 1.0].
Parsing failures are recorded with parsing_success=False and a fallback
score of 0.0, plus a line appended to logs/parsing_errors_{run_id}.jsonl.

Resume support: skips (generation_id, explanation_index, validator_model)
triples already present in validations.jsonl.
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import uuid
from pathlib import Path

# Guards concurrent appends to validations.jsonl, costs.csv, and parse-error log.
_WRITE_LOCK = asyncio.Lock()

from .config import Config, resolve_api_keys
from .generate import read_generations
from .llm_client import FatalLLMError, LLMClient, LLMResponse
from .prep import read_items
from .prompts import build_validation_prompt
from .schemas import GenerationRecord, PrepItem, ValidationRecord

PROGRESS_INTERVAL = 100
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _existing_triples(validations_path: Path) -> set[tuple[str, str, str, str]]:
    """Return set of (item_id, candidate_sense, explanation_text, validator_model) done.

    Keying on stable content (item/sense/explanation/validator) rather than
    generation_id, so resume still works if Stage 2 was re-run and assigned
    fresh generation_ids to the same explanation text.
    """
    done: set[tuple[str, str, str, str]] = set()
    if not validations_path.exists():
        return done
    with validations_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(
                    (
                        rec["item_id"],
                        rec["candidate_sense"],
                        rec["explanation_text"],
                        rec["validator_model"],
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def parse_validity_score(text: str) -> tuple[float, bool]:
    """Parse the validator response. Returns (score, parsing_success).

    On parsing failure, returns (0.0, False).
    """
    match = FLOAT_RE.search(text or "")
    if not match:
        return 0.0, False
    try:
        score = float(match.group(0))
    except ValueError:
        return 0.0, False
    # Heuristic: only normalize obvious 0-10 scale values (>= 2.0).
    # Values in [1.0, 2.0) are treated as clamping errors and clamped to 1.0,
    # since a 0-10 model would not produce 1.2 to mean "low probability".
    if score >= 2.0 and score <= 10.0:
        score = score / 10.0
    score = max(0.0, min(1.0, score))
    return score, True


def _log_parsing_error(
    log_path: Path, generation_id: str, validator: str, raw_response: str
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "generation_id": generation_id,
                    "validator": validator,
                    "raw_response": raw_response[:500],
                }
            )
            + "\n"
        )


def _append_cost(costs_csv: Path, stage: str, model: str, item_id: str, resp: LLMResponse) -> None:
    is_new = not costs_csv.exists()
    costs_csv.parent.mkdir(parents=True, exist_ok=True)
    with costs_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(
                [
                    "stage",
                    "provider",
                    "model",
                    "item_id",
                    "input_tokens",
                    "output_tokens",
                    "cost_usd",
                    "latency_ms",
                    "timestamp",
                ]
            )
        provider = model.split(":", 1)[0]
        w.writerow(
            [
                stage,
                provider,
                model,
                item_id,
                resp.input_tokens,
                resp.output_tokens,
                f"{resp.cost_usd:.8f}",
                resp.latency_ms,
                resp.timestamp,
            ]
        )


async def _validate_one(
    *,
    client: LLMClient,
    item: PrepItem,
    gen: GenerationRecord,
    explanation: str,
    validator_model: str,
    config: Config,
    definitions_path: str,
    out_path: Path,
    costs_csv: Path,
    parse_log: Path,
) -> ValidationRecord | None:
    system, user = build_validation_prompt(
        gen.candidate_sense, item.arg1, item.arg2, explanation, definitions_path
    )
    try:
        resp = await client.call(
            validator_model,
            system=system,
            user=user,
            temperature=0.0,
            max_tokens=config.pipeline.max_tokens_validate,
            json_mode=False,
            stage="validate",
            item_id=item.item_id,
        )
    except FatalLLMError as e:
        print(f"  FATAL [{validator_model}] item={item.item_id}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR [{validator_model}] item={item.item_id}: {e}")
        return None

    score, ok = parse_validity_score(resp.response_text)

    record = ValidationRecord(
        validation_id=str(uuid.uuid4()),
        generation_id=gen.generation_id,
        item_id=item.item_id,
        candidate_sense=gen.candidate_sense,
        validator_model=validator_model,
        explanation_text=explanation,
        validity_score=score,
        raw_response=resp.raw_response,
        parsing_success=ok,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_ms=resp.latency_ms,
        timestamp=resp.timestamp,
    )
    async with _WRITE_LOCK:
        if not ok:
            _log_parsing_error(parse_log, gen.generation_id, validator_model, resp.response_text)
        with out_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        _append_cost(costs_csv, "validate", validator_model, item.item_id, resp)
    return record


async def run_validate(config: Config, run_dir: Path, project_root: Path) -> Path:
    """Run Stage 3. Reads generations.jsonl + items.jsonl, writes validations.jsonl."""
    items_path = run_dir / "items.jsonl"
    gens_path = run_dir / "generations.jsonl"
    out_path = run_dir / "validations.jsonl"
    costs_csv = run_dir / "costs.csv"
    log_path = Path(config.data.logs_dir) / f"llm_calls_{run_dir.name}.jsonl"
    parse_log = Path(config.data.logs_dir) / f"parsing_errors_{run_dir.name}.jsonl"

    items_by_id = {it.item_id: it for it in read_items(items_path)}
    gens = read_generations(gens_path)
    done = _existing_triples(out_path)
    definitions_path = str(project_root / config.data.sense_definitions)

    api_keys = resolve_api_keys()
    client = LLMClient(
        api_keys=api_keys,
        concurrency_per_provider=config.pipeline.concurrency_per_provider,
        max_retries=config.pipeline.max_retries,
        log_path=project_root / log_path,
    )

    tasks: list[asyncio.Task] = []
    for gen in gens:
        if gen.abstained or not gen.explanations:
            continue
        item = items_by_id.get(gen.item_id)
        if item is None:
            continue
        for explanation in gen.explanations:
            for validator_model in config.models.validators:
                if (gen.item_id, gen.candidate_sense, explanation, validator_model) in done:
                    continue
                tasks.append(
                    asyncio.create_task(
                        _validate_one(
                            client=client,
                            item=item,
                            gen=gen,
                            explanation=explanation,
                            validator_model=validator_model,
                            config=config,
                            definitions_path=definitions_path,
                            out_path=out_path,
                            costs_csv=costs_csv,
                            parse_log=project_root / parse_log,
                        )
                    )
                )

    total = len(tasks)
    print(f"[Stage 3] {total} validation calls to make ({len(done)} already done)")
    completed = 0
    running_cost = 0.0
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        completed += 1
        if rec is not None:
            running_cost += rec.cost_usd
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            print(f"  [Stage 3] {completed}/{total} done, running cost ${running_cost:.4f}")

    print(f"[Stage 3] Wrote {out_path}, total cost ${running_cost:.4f}")
    return out_path


def read_validations(validations_jsonl: Path) -> list[ValidationRecord]:
    """Read validations.jsonl back, validating each row."""
    records: list[ValidationRecord] = []
    with validations_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ValidationRecord(**json.loads(line)))
    return records
