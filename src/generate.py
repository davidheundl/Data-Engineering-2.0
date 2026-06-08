"""Stage 2: Generate — produce LLM explanations for each (item, sense, model).

Reads items.jsonl. For each PrepItem, iterates over its candidate_senses
(senses with >= 1 crowd vote) and each generator_model. One async LLM call
per triple. Parses the JSON response defensively and validates against
``GenerationRecord``.

Resume support: skips (item_id, candidate_sense, generator_model) triples
already present in generations.jsonl.
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import uuid
from pathlib import Path

# Guards concurrent appends to generations.jsonl and costs.csv from many tasks.
_WRITE_LOCK = asyncio.Lock()

from .config import Config, resolve_api_keys
from .llm_client import FatalLLMError, LLMClient, LLMResponse
from .prep import read_items
from .prompts import build_generation_prompt, get_enabled_senses
from .schemas import GenerationRecord, LEVEL2_SENSES_NO_NOREL, PrepItem

PROGRESS_INTERVAL = 50


def _existing_triples(generations_path: Path) -> set[tuple[str, str, str]]:
    """Return set of (item_id, candidate_sense, generator_model) already done."""
    done: set[tuple[str, str, str]] = set()
    if not generations_path.exists():
        return done
    with generations_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["item_id"], rec["candidate_sense"], rec["generator_model"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _parse_generation_response(text: str) -> tuple[list[str], bool, str | None]:
    """Parse a Stage-2 LLM response into (explanations, abstained, reason).

    The prompt now forbids abstention — the model must always return at least
    one explanation. Abstention is only used as a fallback for parse failures
    or completely empty model outputs.
    """
    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    # Find the outermost JSON object
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        # As a last-ditch fallback, treat the entire response text as a single
        # explanation rather than abstaining — the model produced *something*.
        stripped = text.strip()
        if stripped:
            return [stripped[:2000]], False, None
        return [], True, "model returned empty response"

    payload = candidate[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        # Same fallback: keep whatever text the model produced.
        stripped = text.strip()
        if stripped:
            return [stripped[:2000]], False, None
        return [], True, f"json error: {e}"

    # Support both single ("explanation": "...") and legacy list ("explanations": [...])
    single = data.get("explanation")
    if single and isinstance(single, str) and single.strip():
        return [single.strip()], False, None

    explanations = data.get("explanations") or []
    if not isinstance(explanations, list):
        explanations = []
    explanations = [str(x).strip() for x in explanations if str(x).strip()]

    abstained = len(explanations) == 0
    reason = "model returned empty explanation" if abstained else None
    return explanations[:1] if explanations else [], abstained, reason


def _append_cost(costs_csv: Path, stage: str, model: str, item_id: str, resp: LLMResponse) -> None:
    """Append one row to costs.csv (created with header if missing)."""
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


async def _generate_one(
    *,
    client: LLMClient,
    item: PrepItem,
    sense: str,
    model: str,
    config: Config,
    definitions_path: str,
    out_path: Path,
    costs_csv: Path,
) -> GenerationRecord | None:
    """One LLM call for (item, sense, model). Returns the record or None on fatal error."""
    system, user = build_generation_prompt(sense, item.arg1, item.arg2, definitions_path)
    try:
        resp = await client.call(
            model,
            system=system,
            user=user,
            temperature=config.pipeline.temperature_generate,
            max_tokens=config.pipeline.max_tokens_generate,
            json_mode=True,
            stage="generate",
            item_id=item.item_id,
        )
    except FatalLLMError as e:
        print(f"  FATAL [{model}] item={item.item_id} sense={sense}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR [{model}] item={item.item_id} sense={sense}: {e}")
        return None

    explanations, abstained, reason = _parse_generation_response(resp.response_text)
    record = GenerationRecord(
        generation_id=str(uuid.uuid4()),
        item_id=item.item_id,
        candidate_sense=sense,
        generator_model=model,
        explanations=explanations,
        abstained=abstained,
        abstention_reason=reason,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_ms=resp.latency_ms,
        timestamp=resp.timestamp,
        raw_response=resp.raw_response,
    )

    # Append to file immediately for resume support. The lock prevents
    # interleaved writes from concurrent tasks corrupting JSONL lines.
    async with _WRITE_LOCK:
        with out_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        _append_cost(costs_csv, "generate", model, item.item_id, resp)
    return record


async def run_generate(config: Config, run_dir: Path, project_root: Path) -> Path:
    """Run Stage 2. Reads items.jsonl, writes generations.jsonl."""
    items_path = run_dir / "items.jsonl"
    out_path = run_dir / "generations.jsonl"
    costs_csv = run_dir / "costs.csv"
    log_path = Path(config.data.logs_dir) / f"llm_calls_{run_dir.name}.jsonl"

    items = read_items(items_path)
    done = _existing_triples(out_path)
    definitions_path = str(project_root / config.data.sense_definitions)

    api_keys = resolve_api_keys()
    client = LLMClient(
        api_keys=api_keys,
        concurrency_per_provider=config.pipeline.concurrency_per_provider,
        max_retries=config.pipeline.max_retries,
        log_path=project_root / log_path,
    )

    # Use enabled senses from definitions file (respects filled=true/false)
    enabled_senses = get_enabled_senses(definitions_path)
    print(f"[Stage 2] Using {len(enabled_senses)} enabled senses: {enabled_senses}")

    # Build task list
    tasks: list[asyncio.Task] = []
    for item in items:
        for sense in enabled_senses:
            for model in config.models.generators:
                if (item.item_id, sense, model) in done:
                    continue
                tasks.append(
                    asyncio.create_task(
                        _generate_one(
                            client=client,
                            item=item,
                            sense=sense,
                            model=model,
                            config=config,
                            definitions_path=definitions_path,
                            out_path=out_path,
                            costs_csv=costs_csv,
                        )
                    )
                )

    total = len(tasks)
    print(f"[Stage 2] {total} generation calls to make ({len(done)} already done)")
    completed = 0
    running_cost = 0.0
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        completed += 1
        if rec is not None:
            running_cost += rec.cost_usd
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            print(f"  [Stage 2] {completed}/{total} done, running cost ${running_cost:.4f}")

    print(f"[Stage 2] Wrote {out_path}, total cost ${running_cost:.4f}")
    return out_path


def read_generations(generations_jsonl: Path) -> list[GenerationRecord]:
    """Read generations.jsonl back, validating each row."""
    records: list[GenerationRecord] = []
    with generations_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(GenerationRecord(**json.loads(line)))
    return records
