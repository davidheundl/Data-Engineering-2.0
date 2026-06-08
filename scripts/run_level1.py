#!/usr/bin/env python3
"""Level-1 EVADE-style pipeline.

Standalone script that runs 5 stages on Level-1 PDTB senses (Temporal,
Contingency, Comparison, Expansion) with EVADE-style generation (multiple
explanations per sense, abstention allowed).

Reuses existing modules: llm_client, schemas, config, prompts, aggregate logic,
analyze.

Usage:
    python scripts/run_level1.py --config configs/level1_experiment.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.aggregate import _aggregate_one_item, _kld, _build_llm_distribution
from src.analyze import run_analyze
from src.config import Config, load_config, resolve_api_keys
from src.llm_client import FatalLLMError, LLMClient, LLMResponse
from src.pipeline import make_run_id
from src.prompts import load_sense_definitions, _lookup_sense
from src.schemas import (
    DistributionRecord,
    GenerationRecord,
    PrepItem,
    RunMetrics,
    SenseStats,
    ValidationRecord,
)
from src.validate import parse_validity_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEVEL1_SENSES = ["temporal", "contingency", "comparison", "expansion"]
GENRE_MAP = {"novel": "Lit", "europarl": "Europarl", "wikipedia": "Wiki"}
AGREEMENT_HIGH_THRESHOLD = 0.5
PROGRESS_INTERVAL = 50

# Guards concurrent appends to JSONL files.
_WRITE_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Stage 1: Prep (Level-1)
# ---------------------------------------------------------------------------
def _split_high_low(quota: int) -> tuple[int, int]:
    high = math.ceil(quota / 2)
    return high, quota - high


def _build_crowd_distribution(senses: list[str]) -> dict[str, float]:
    counts = Counter(senses)
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()}


def prep_l1(config: Config, run_dir: Path, project_root: Path, *, items_file: str | None = None) -> Path:
    """Stage 1: Build items.jsonl from DiscoGeM using lev1_conn2.

    If items_file is provided, select exactly those item IDs (no sampling).
    """
    import pandas as pd

    wide_path = project_root / config.data.wide_csv
    full_path = project_root / config.data.full_csv

    wide = pd.read_csv(wide_path, low_memory=False)
    full = pd.read_csv(full_path, low_memory=False)

    # Filter to three genres
    wide = wide[wide["genre"].isin(GENRE_MAP.keys())].copy()
    full = full[full["genre"].isin(GENRE_MAP.keys())].copy()

    # Drop NA Level-1 senses
    full = full[full["lev1_conn2"].notna()].copy()
    full = full[full["lev1_conn2"].astype(str).str.upper() != "NA"].copy()

    # Drop multi-valued entries (e.g. "expansion,contingency")
    full = full[~full["lev1_conn2"].astype(str).str.contains(",")].copy()

    # Group per-item annotator senses (Level-1)
    sense_lists = (
        full.groupby("itemid")["lev1_conn2"].apply(list).rename("annotator_senses")
    )
    df = wide.merge(sense_lists, left_on="itemid", right_index=True, how="inner")
    df["genre_mapped"] = df["genre"].map(GENRE_MAP)

    # Build PrepItems
    candidates: list[PrepItem] = []
    for _, row in df.iterrows():
        senses = list(row["annotator_senses"])
        if len(senses) < 2:
            continue
        distribution = _build_crowd_distribution(senses)
        candidate_senses = sorted(s for s in distribution.keys() if s != "norel")
        majority = max(distribution.items(), key=lambda kv: kv[1])[0]
        agreement = max(distribution.values())
        bin_ = "high" if agreement >= AGREEMENT_HIGH_THRESHOLD else "low"

        candidates.append(PrepItem(
            item_id=str(row["itemid"]),
            genre=row["genre_mapped"],
            arg1=str(row["arg1"]),
            arg2=str(row["arg2"]),
            arg1_singlesentence=str(row.get("arg1_singlesentence", "")),
            arg2_singlesentence=str(row.get("arg2_singlesentence", "")),
            annotator_step2_senses=senses,
            n_valid_annotations=len(senses),
            crowd_sense_distribution=distribution,
            candidate_senses=candidate_senses,
            majority_single_sense=majority,
            wikipedia_reference_labels=None,
            crowd_agreement_score=agreement,
            stratification_bin=bin_,
            split=str(row["split"]) if "split" in row and pd.notna(row.get("split")) else None,
        ))

    # Select items: either from file or via stratified sampling
    if items_file:
        with open(items_file) as f:
            wanted = {line.strip() for line in f if line.strip()}
        candidates_by_id = {c.item_id: c for c in candidates}
        sampled = []
        missing = []
        for iid in sorted(wanted):
            if iid in candidates_by_id:
                sampled.append(candidates_by_id[iid])
            else:
                missing.append(iid)
        if missing:
            print(f"  WARNING: {len(missing)} item IDs not found: {missing[:5]}...")
    else:
        import random
        rng = random.Random(config.sampling.seed)
        sampled: list[PrepItem] = []
        for genre, quota in config.sampling.genre_split.items():
            pool = [c for c in candidates if c.genre == genre]
            if quota >= len(pool):
                pool.sort(key=lambda c: c.item_id)
                sampled.extend(pool)
                continue
            high_quota, low_quota = _split_high_low(quota)
            high_pool = sorted([c for c in pool if c.stratification_bin == "high"], key=lambda c: c.item_id)
            low_pool = sorted([c for c in pool if c.stratification_bin == "low"], key=lambda c: c.item_id)
            if len(high_pool) < high_quota:
                raise ValueError(f"Not enough 'high' items for {genre}: need {high_quota}, have {len(high_pool)}")
            if len(low_pool) < low_quota:
                raise ValueError(f"Not enough 'low' items for {genre}: need {low_quota}, have {len(low_pool)}")
            sampled.extend(rng.sample(high_pool, high_quota))
            sampled.extend(rng.sample(low_pool, low_quota))

    out_path = run_dir / "items.jsonl"
    with out_path.open("w") as f:
        for item in sampled:
            f.write(item.model_dump_json() + "\n")

    print(f"[Stage 1] Wrote {len(sampled)} items to {out_path} (from pool of {len(candidates)} candidates)")
    counts: dict[tuple[str, str], int] = {}
    for it in sampled:
        counts[(it.genre, it.stratification_bin)] = counts.get((it.genre, it.stratification_bin), 0) + 1
    for (g, b), n in sorted(counts.items()):
        print(f"  {g:<10s} {b:<5s} -> {n}")
    return out_path


# ---------------------------------------------------------------------------
# Stage 2: Generate (EVADE-style, multiple explanations)
# ---------------------------------------------------------------------------
GENERATION_SYSTEM_L1 = (
    "You are an expert in discourse relation analysis under the "
    "Penn Discourse Treebank 3.0 framework."
)


def build_generation_prompt_l1(
    sense: str, arg1: str, arg2: str, definitions_path: str
) -> tuple[str, str]:
    """Build prompt for EVADE-style generation with multiple explanations."""
    definitions = load_sense_definitions(definitions_path)
    entry = _lookup_sense(definitions, sense)

    user = (
        f"The candidate discourse relation sense is {sense}.\n\n"
        f"Definition: {entry['definition']}\n\n"
        f"Canonical example: {entry['example']}\n\n"
        f"Generate every distinct explanation you can think of for why the "
        f"implicit discourse relation between Argument 1 and Argument 2 below "
        f"could express {sense}.\n\n"
        f"- Each explanation should present a DIFFERENT reasoning or perspective\n"
        f"- Do NOT repeat the same argument in different words\n"
        f"- If you genuinely cannot justify this sense for these arguments, "
        f"return an empty list\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        f"Output strictly valid JSON.\n\n"
        f"Output JSON schema:\n"
        f'{{\n'
        f'  "explanations": ["...", "...", ...]\n'
        f'}}'
    )
    return GENERATION_SYSTEM_L1, user


def _parse_generation_response_l1(text: str) -> tuple[list[str], bool, str | None]:
    """Parse EVADE-style response: multiple explanations, abstention = empty list."""
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        stripped = text.strip()
        if stripped:
            return [stripped[:2000]], False, None
        return [], True, "model returned empty response"

    payload = candidate[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        stripped = text.strip()
        if stripped:
            return [stripped[:2000]], False, None
        return [], True, f"json error: {e}"

    # Support both formats
    single = data.get("explanation")
    if single and isinstance(single, str) and single.strip():
        return [single.strip()], False, None

    explanations = data.get("explanations") or []
    if not isinstance(explanations, list):
        explanations = []
    explanations = [str(x).strip() for x in explanations if str(x).strip()]

    # EVADE: keep ALL explanations (no [:1] truncation)
    abstained = len(explanations) == 0
    reason = "model returned empty list (abstained)" if abstained else None
    return explanations, abstained, reason


def _append_cost(costs_csv: Path, stage: str, model: str, item_id: str, resp: LLMResponse) -> None:
    is_new = not costs_csv.exists()
    costs_csv.parent.mkdir(parents=True, exist_ok=True)
    with costs_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["stage", "provider", "model", "item_id", "input_tokens",
                         "output_tokens", "cost_usd", "latency_ms", "timestamp"])
        provider = model.split(":", 1)[0]
        w.writerow([stage, provider, model, item_id, resp.input_tokens,
                     resp.output_tokens, f"{resp.cost_usd:.8f}", resp.latency_ms, resp.timestamp])


async def _generate_one_l1(
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
    system, user = build_generation_prompt_l1(sense, item.arg1, item.arg2, definitions_path)
    try:
        resp = await client.call(
            model, system=system, user=user,
            temperature=config.pipeline.temperature_generate,
            max_tokens=config.pipeline.max_tokens_generate,
            json_mode=True, stage="generate", item_id=item.item_id,
        )
    except FatalLLMError as e:
        print(f"  FATAL [{model}] item={item.item_id} sense={sense}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR [{model}] item={item.item_id} sense={sense}: {e}")
        return None

    explanations, abstained, reason = _parse_generation_response_l1(resp.response_text)
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
    async with _WRITE_LOCK:
        with out_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        _append_cost(costs_csv, "generate", model, item.item_id, resp)
    return record


def _existing_gen_triples(path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not path.exists():
        return done
    with path.open("r") as f:
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


async def generate_l1(config: Config, run_dir: Path, project_root: Path) -> Path:
    """Stage 2: EVADE-style generation over 4 Level-1 senses."""
    items_path = run_dir / "items.jsonl"
    out_path = run_dir / "generations.jsonl"
    costs_csv = run_dir / "costs.csv"
    log_path = Path(config.data.logs_dir) / f"llm_calls_{run_dir.name}.jsonl"

    items = _read_items(items_path)
    done = _existing_gen_triples(out_path)
    definitions_path = str(project_root / config.data.sense_definitions)

    api_keys = resolve_api_keys()
    client = LLMClient(
        api_keys=api_keys,
        concurrency_per_provider=config.pipeline.concurrency_per_provider,
        max_retries=config.pipeline.max_retries,
        log_path=project_root / log_path,
    )

    print(f"[Stage 2] Using {len(LEVEL1_SENSES)} Level-1 senses: {LEVEL1_SENSES}")

    tasks: list[asyncio.Task] = []
    for item in items:
        for sense in LEVEL1_SENSES:
            for model in config.models.generators:
                if (item.item_id, sense, model) in done:
                    continue
                tasks.append(asyncio.create_task(
                    _generate_one_l1(
                        client=client, item=item, sense=sense, model=model,
                        config=config, definitions_path=definitions_path,
                        out_path=out_path, costs_csv=costs_csv,
                    )
                ))

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


# ---------------------------------------------------------------------------
# Stage 3: Validate (cross-validation, reuses existing logic)
# ---------------------------------------------------------------------------
def _build_validation_prompt_l1(
    sense: str, arg1: str, arg2: str, explanation: str, definitions_path: str
) -> tuple[str, str]:
    """Validation prompt for Level-1 senses."""
    definitions = load_sense_definitions(definitions_path)
    entry = _lookup_sense(definitions, sense)

    system = "You are an expert linguistic annotator under the PDTB 3.0 framework."
    user = (
        f"You are evaluating whether an explanation correctly justifies "
        f"assigning a specific discourse relation label to a pair of text "
        f"arguments.\n\n"
        f"Discourse relation sense: {sense}\n"
        f"Definition: {entry['definition']}\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        f"Explanation for why this pair expresses '{sense}': {explanation}\n\n"
        f"Rate the explanation on a 0\u201310 integer scale:\n"
        f"  0\u20132: The explanation is poor \u2014 it does not convincingly justify "
        f"'{sense}' for this argument pair, or the relation clearly does not "
        f"hold.\n"
        f"  3\u20135: The explanation is mediocre \u2014 there is a weak or partial "
        f"connection, but it is not compelling.\n"
        f"  6\u20137: The explanation is reasonable \u2014 it makes a fair case for "
        f"'{sense}', though some aspects could be stronger.\n"
        f"  8\u201310: The explanation is strong \u2014 it clearly and convincingly "
        f"justifies '{sense}' for this specific argument pair.\n\n"
        f"Be genuinely critical. If the discourse relation does not fit this "
        f"argument pair well, the explanation cannot be good regardless of how "
        f"well-written it is. Give ONLY a single integer (0\u201310), no other "
        f"text.\n\n"
        f"Score:"
    )
    return system, user


def _existing_val_triples(path: Path) -> set[tuple[str, str, str, str]]:
    done: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return done
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["item_id"], rec["candidate_sense"],
                           rec["explanation_text"], rec["validator_model"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _log_parsing_error(log_path: Path, generation_id: str, validator: str, raw_response: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({"generation_id": generation_id, "validator": validator,
                             "raw_response": raw_response[:500]}) + "\n")


async def _validate_one_l1(
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
    system, user = _build_validation_prompt_l1(
        gen.candidate_sense, item.arg1, item.arg2, explanation, definitions_path
    )
    try:
        resp = await client.call(
            validator_model, system=system, user=user,
            temperature=0.0, max_tokens=config.pipeline.max_tokens_validate,
            json_mode=False, stage="validate", item_id=item.item_id,
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


async def validate_l1(config: Config, run_dir: Path, project_root: Path) -> Path:
    """Stage 3: Cross-validation of all explanations."""
    items_path = run_dir / "items.jsonl"
    gens_path = run_dir / "generations.jsonl"
    out_path = run_dir / "validations.jsonl"
    costs_csv = run_dir / "costs.csv"
    log_path = Path(config.data.logs_dir) / f"llm_calls_{run_dir.name}.jsonl"
    parse_log = Path(config.data.logs_dir) / f"parsing_errors_{run_dir.name}.jsonl"

    items_by_id = {it.item_id: it for it in _read_items(items_path)}
    gens = _read_generations(gens_path)
    done = _existing_val_triples(out_path)
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
                if validator_model == gen.generator_model:
                    continue
                if (gen.item_id, gen.candidate_sense, explanation, validator_model) in done:
                    continue
                tasks.append(asyncio.create_task(
                    _validate_one_l1(
                        client=client, item=item, gen=gen, explanation=explanation,
                        validator_model=validator_model, config=config,
                        definitions_path=definitions_path, out_path=out_path,
                        costs_csv=costs_csv, parse_log=project_root / parse_log,
                    )
                ))

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


# ---------------------------------------------------------------------------
# Stage 4: Compare — select best explanation per sense, then comparative scoring
# ---------------------------------------------------------------------------
COMPARE_SYSTEM = (
    "You are an expert linguistic annotator under the PDTB 3.0 framework."
)


def _select_best_explanations(
    item_id: str,
    gens: list[GenerationRecord],
    vals: list[ValidationRecord],
) -> dict[str, str]:
    """For each sense, pick the explanation with the highest mean validation score.

    Returns {sense: best_explanation_text}. Senses with no explanations are omitted.
    """
    # Group validation scores by (sense, explanation_text)
    scores_by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for v in vals:
        if v.item_id == item_id:
            scores_by[(v.candidate_sense, v.explanation_text)].append(v.validity_score)

    # Collect all explanations per sense from generations
    expls_by_sense: dict[str, set[str]] = defaultdict(set)
    for g in gens:
        if g.item_id == item_id and not g.abstained:
            for e in g.explanations:
                expls_by_sense[g.candidate_sense].add(e)

    best: dict[str, str] = {}
    for sense in LEVEL1_SENSES:
        candidates = expls_by_sense.get(sense, set())
        if not candidates:
            continue
        # Pick the explanation with the highest mean validation score
        best_expl = None
        best_score = -1.0
        for expl in candidates:
            scores = scores_by.get((sense, expl), [])
            m = mean(scores) if scores else 0.0
            if m > best_score:
                best_score = m
                best_expl = expl
        if best_expl is not None:
            best[sense] = best_expl
    return best


def _build_compare_prompt(
    arg1: str, arg2: str, best_explanations: dict[str, str], definitions_path: str
) -> tuple[str, str]:
    """Build comparative prompt WITH explanations: distribute 100 points."""
    definitions = load_sense_definitions(definitions_path)

    sense_blocks = []
    for sense in LEVEL1_SENSES:
        entry = definitions.get(sense, {})
        definition = entry.get("definition", "")
        expl = best_explanations.get(sense)
        if expl:
            sense_blocks.append(
                f"**{sense}**\n"
                f"Definition: {definition}\n"
                f"Best explanation: {expl}"
            )
        else:
            sense_blocks.append(
                f"**{sense}**\n"
                f"Definition: {definition}\n"
                f"Best explanation: (no explanation was generated — all models abstained)"
            )

    user = (
        f"For the following pair of text arguments, you are given the best "
        f"available explanation for each of 4 discourse relation senses. "
        f"Your task is to judge how well each sense fits this specific "
        f"argument pair.\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        + "\n\n".join(sense_blocks) + "\n\n"
        f"Distribute exactly 100 points across the 4 senses. A higher score "
        f"means the sense is a better fit for this argument pair. Consider "
        f"both the explanation quality AND whether the sense itself is "
        f"appropriate. If a sense clearly does not fit, give it 0-5 points.\n\n"
        f"Output strictly valid JSON with exactly these 4 keys:\n"
        f'{{"temporal": X, "contingency": Y, "comparison": Z, "expansion": W}}'
    )
    return COMPARE_SYSTEM, user


def _build_compare_prompt_no_explanations(
    arg1: str, arg2: str, definitions_path: str
) -> tuple[str, str]:
    """Build comparative prompt WITHOUT explanations (ablation)."""
    definitions = load_sense_definitions(definitions_path)

    sense_blocks = []
    for sense in LEVEL1_SENSES:
        entry = definitions.get(sense, {})
        definition = entry.get("definition", "")
        sense_blocks.append(
            f"**{sense}**\n"
            f"Definition: {definition}"
        )

    user = (
        f"For the following pair of text arguments, judge how well each of "
        f"4 discourse relation senses fits.\n\n"
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n\n"
        + "\n\n".join(sense_blocks) + "\n\n"
        f"Distribute exactly 100 points across the 4 senses. A higher score "
        f"means the sense is a better fit for this argument pair. "
        f"If a sense clearly does not fit, give it 0-5 points.\n\n"
        f"Output strictly valid JSON with exactly these 4 keys:\n"
        f'{{"temporal": X, "contingency": Y, "comparison": Z, "expansion": W}}'
    )
    return COMPARE_SYSTEM, user


def _parse_compare_response(text: str) -> dict[str, float] | None:
    """Parse a comparative response into a normalized distribution."""
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None

    dist: dict[str, float] = {}
    for sense in LEVEL1_SENSES:
        val = data.get(sense, 0)
        try:
            dist[sense] = float(val)
        except (TypeError, ValueError):
            dist[sense] = 0.0

    total = sum(dist.values())
    if total <= 0:
        return None
    # Normalize to sum to 1.0
    return {s: v / total for s, v in dist.items()}


async def _compare_one(
    *,
    client: LLMClient,
    item: PrepItem,
    model: str,
    best_explanations: dict[str, str] | None,
    definitions_path: str,
    costs_csv: Path,
) -> tuple[str, str, dict[str, float] | None]:
    """One comparative LLM call. Returns (item_id, model, distribution_or_None).

    If best_explanations is None, uses the no-explanations ablation prompt.
    """
    if best_explanations is not None:
        system, user = _build_compare_prompt(
            item.arg1, item.arg2, best_explanations, definitions_path
        )
    else:
        system, user = _build_compare_prompt_no_explanations(
            item.arg1, item.arg2, definitions_path
        )
    try:
        resp = await client.call(
            model, system=system, user=user,
            temperature=0.0, max_tokens=200,
            json_mode=True, stage="compare", item_id=item.item_id,
        )
    except FatalLLMError as e:
        print(f"  FATAL [{model}] item={item.item_id}: {e}")
        return item.item_id, model, None
    except Exception as e:
        print(f"  ERROR [{model}] item={item.item_id}: {e}")
        return item.item_id, model, None

    async with _WRITE_LOCK:
        _append_cost(costs_csv, "compare", model, item.item_id, resp)

    dist = _parse_compare_response(resp.response_text)
    if dist is None:
        print(f"  PARSE FAIL [{model}] item={item.item_id}: {resp.response_text[:100]}")
    return item.item_id, model, dist


async def compare_l1(
    config: Config, run_dir: Path, project_root: Path, *, no_explanations: bool = False
) -> Path:
    """Stage 4: Comparative scoring — each model distributes 100 points.

    If no_explanations=True, runs the ablation variant (definitions only, no EVADE explanations).
    """
    mode = "NO-EXPLANATIONS (ablation)" if no_explanations else "WITH explanations"
    items = _read_items(run_dir / "items.jsonl")
    costs_csv = run_dir / "costs.csv"
    log_path = Path(config.data.logs_dir) / f"llm_calls_{run_dir.name}.jsonl"
    definitions_path = str(project_root / config.data.sense_definitions)

    api_keys = resolve_api_keys()
    client = LLMClient(
        api_keys=api_keys,
        concurrency_per_provider=config.pipeline.concurrency_per_provider,
        max_retries=config.pipeline.max_retries,
        log_path=project_root / log_path,
    )

    # Select best explanation per (item, sense) — only if using explanations
    best_per_item: dict[str, dict[str, str] | None] = {}
    if not no_explanations:
        gens = _read_generations(run_dir / "generations.jsonl")
        vals = _read_validations(run_dir / "validations.jsonl")
        for item in items:
            best_per_item[item.item_id] = _select_best_explanations(
                item.item_id, gens, vals
            )
    else:
        for item in items:
            best_per_item[item.item_id] = None

    # Launch comparative calls: each model scores each item
    tasks: list[asyncio.Task] = []
    for item in items:
        for model in config.models.validators:
            tasks.append(asyncio.create_task(
                _compare_one(
                    client=client, item=item, model=model,
                    best_explanations=best_per_item[item.item_id],
                    definitions_path=definitions_path, costs_csv=costs_csv,
                )
            ))

    total = len(tasks)
    print(f"[Stage 4] {total} comparative calls to make ({mode})")
    results: list[tuple[str, str, dict[str, float] | None]] = []
    completed = 0
    for fut in asyncio.as_completed(tasks):
        result = await fut
        results.append(result)
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"  [Stage 4] {completed}/{total} done")

    # Average distributions across models per item
    per_item_dists: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item_id, model, dist in results:
        if dist is not None:
            per_item_dists[item_id].append(dist)

    # Write comparative_results.jsonl (raw per-model results)
    comp_path = run_dir / "comparative_results.jsonl"
    with comp_path.open("w") as f:
        for item_id, model, dist in results:
            f.write(json.dumps({
                "item_id": item_id, "model": model,
                "distribution": dist,
                "best_explanations": best_per_item[item_id],
            }) + "\n")

    # Build averaged LLM distribution per item
    avg_dists: dict[str, dict[str, float]] = {}
    for item_id, model_dists in per_item_dists.items():
        if not model_dists:
            avg_dists[item_id] = {}
            continue
        merged: dict[str, float] = defaultdict(float)
        for d in model_dists:
            for s, v in d.items():
                merged[s] += v
        n = len(model_dists)
        avg_dists[item_id] = {s: v / n for s, v in merged.items()}

    # Build distributions.jsonl (compatible with analyze)
    # per_sense_stats from original validation data (if available), llm_dist from comparative
    gens_path = run_dir / "generations.jsonl"
    vals_path = run_dir / "validations.jsonl"
    has_evade_data = gens_path.exists() and vals_path.exists()

    gens_by_item: dict[str, list[GenerationRecord]] = defaultdict(list)
    vals_by_item: dict[str, list[ValidationRecord]] = defaultdict(list)
    if has_evade_data:
        for g in _read_generations(gens_path):
            gens_by_item[g.item_id].append(g)
        for v in _read_validations(vals_path):
            vals_by_item[v.item_id].append(v)

    dist_path = run_dir / "distributions.jsonl"
    records: list[DistributionRecord] = []
    with dist_path.open("w") as f:
        for item in items:
            if has_evade_data:
                _, sense_stats = _aggregate_one_item(
                    item, gens_by_item.get(item.item_id, []),
                    vals_by_item.get(item.item_id, []), config,
                    enabled_senses=LEVEL1_SENSES,
                )
            else:
                # No EVADE data — fill with empty stats
                sense_stats = {
                    s: SenseStats(max_validity=0, mean_validity=0, n_explanations=0,
                                  n_generators_abstained=0, validator_std=0, commission_score=0)
                    for s in LEVEL1_SENSES
                }
            # Use comparative distribution for ALL tau values
            comp_dist = avg_dists.get(item.item_id, {})
            llm_dist_per_tau = {
                f"{tau:.2f}": comp_dist
                for tau in config.pipeline.tau_values()
            }
            rec = DistributionRecord(
                item_id=item.item_id,
                candidate_senses=item.candidate_senses,
                per_sense_stats=sense_stats,
                llm_label_distribution_per_tau=llm_dist_per_tau,
            )
            f.write(rec.model_dump_json() + "\n")
            records.append(rec)

    print(f"[Stage 4] Wrote {comp_path} and {dist_path}")

    # Print summary
    for item in items:
        d = avg_dists.get(item.item_id, {})
        crowd = item.crowd_sense_distribution
        if d:
            top = max(d.items(), key=lambda x: x[1])
            print(f"  {item.item_id}: LLM top={top[0]}({top[1]:.2f}) crowd_majority={item.majority_single_sense}")
    return dist_path


# ---------------------------------------------------------------------------
# Stage 4a: Aggregate — softmax over validation scores (original EVADE approach)
# ---------------------------------------------------------------------------
def aggregate_l1(config: Config, run_dir: Path, project_root: Path) -> Path:
    """Aggregate using softmax over mean_validity per sense (tau sweep)."""
    items = _read_items(run_dir / "items.jsonl")
    gens = _read_generations(run_dir / "generations.jsonl")
    vals = _read_validations(run_dir / "validations.jsonl")

    print(f"[Aggregate] Using {len(LEVEL1_SENSES)} Level-1 senses: {LEVEL1_SENSES}")

    gens_by_item: dict[str, list[GenerationRecord]] = defaultdict(list)
    for g in gens:
        gens_by_item[g.item_id].append(g)
    vals_by_item: dict[str, list[ValidationRecord]] = defaultdict(list)
    for v in vals:
        vals_by_item[v.item_id].append(v)

    dist_path = run_dir / "distributions_aggregate.jsonl"
    records: list[DistributionRecord] = []
    with dist_path.open("w") as f:
        for item in items:
            rec, _ = _aggregate_one_item(
                item, gens_by_item.get(item.item_id, []),
                vals_by_item.get(item.item_id, []), config,
                enabled_senses=LEVEL1_SENSES,
            )
            f.write(rec.model_dump_json() + "\n")
            records.append(rec)

    print(f"[Aggregate] Wrote {dist_path}")
    return dist_path


# ---------------------------------------------------------------------------
# Stage 5: Metrics (write metrics.json)
# ---------------------------------------------------------------------------
def write_metrics_l1(config: Config, run_dir: Path) -> Path:
    """Write metrics.json from distributions + raw data."""
    items = _read_items(run_dir / "items.jsonl")
    gens = _read_generations(run_dir / "generations.jsonl")
    vals = _read_validations(run_dir / "validations.jsonl")

    items_by_id = {it.item_id: it for it in items}
    item_genre: dict[str, str] = {it.item_id: it.genre for it in items}

    dist_path = run_dir / "distributions.jsonl"
    records: list[DistributionRecord] = []
    with dist_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(DistributionRecord(**json.loads(line)))

    total_gen_calls = len(gens)
    total_val_calls = len(vals)
    total_cost = sum(g.cost_usd for g in gens) + sum(v.cost_usd for v in vals)
    mean_cost_per_item = total_cost / max(1, len(items))

    gen_counts: dict[str, int] = defaultdict(int)
    abs_counts: dict[str, int] = defaultdict(int)
    for g in gens:
        gen_counts[g.generator_model] += 1
        if g.abstained or not g.explanations:
            abs_counts[g.generator_model] += 1
    abstention_rate = {
        m: (abs_counts[m] / gen_counts[m]) if gen_counts[m] else 0.0
        for m in config.models.generators
    }

    genre_scores: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if not rec.per_sense_stats:
            continue
        max_commission = max(st.commission_score for st in rec.per_sense_stats.values())
        genre_scores[item_genre[rec.item_id]].append(max_commission)
    avg_commission_by_genre = {
        g: (mean(scores) if scores else 0.0) for g, scores in genre_scores.items()
    }

    kld_per_tau: dict[str, float] = {}
    for tau in config.pipeline.tau_values():
        key = f"{tau:.2f}"
        klds: list[float] = []
        for rec in records:
            llm_dist = rec.llm_label_distribution_per_tau.get(key, {})
            if not llm_dist:
                continue
            item = items_by_id[rec.item_id]
            klds.append(_kld(item.crowd_sense_distribution, llm_dist))
        kld_per_tau[key] = mean(klds) if klds else float("nan")

    metrics = RunMetrics(
        run_id=run_dir.name,
        config_name=config.name,
        n_items=len(items),
        total_generation_calls=total_gen_calls,
        total_validation_calls=total_val_calls,
        total_cost_usd=total_cost,
        mean_cost_per_item=mean_cost_per_item,
        abstention_rate_per_model=abstention_rate,
        avg_commission_score_by_genre=avg_commission_by_genre,
        kld_vs_crowd_per_tau=kld_per_tau,
        tau_values=config.pipeline.tau_values(),
    )

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        f.write(metrics.model_dump_json(indent=2))

    print(f"[Stage 5] Wrote {metrics_path}")
    print(f"  KLD per tau: {kld_per_tau}")
    return metrics_path


# ---------------------------------------------------------------------------
# Helpers: read JSONL files
# ---------------------------------------------------------------------------
def _read_items(path: Path) -> list[PrepItem]:
    items: list[PrepItem] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(PrepItem(**json.loads(line)))
    return items


def _read_generations(path: Path) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(GenerationRecord(**json.loads(line)))
    return records


def _read_validations(path: Path) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ValidationRecord(**json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="Level-1 EVADE-style pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stages", default="prep,generate,validate,aggregate,compare,analyze",
                        help="Comma-separated stages to run (default: all)")
    parser.add_argument("--run-id", default=None, help="Resume into existing run dir")
    parser.add_argument("--items-file", default=None, help="Text file with item IDs (one per line), skip sampling")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = PROJECT_ROOT
    stages = [s.strip() for s in args.stages.split(",")]

    import shutil
    if args.run_id:
        run_dir = project_root / config.data.results_dir / args.run_id
        if not run_dir.exists():
            print(f"ERROR: run dir {run_dir} not found")
            sys.exit(1)
    else:
        run_id = make_run_id(config.name, project_root)
        run_dir = project_root / config.data.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, run_dir / "config.yaml")
        senses_src = project_root / config.data.sense_definitions
        if senses_src.exists():
            shutil.copy2(senses_src, run_dir / senses_src.name)

    print(f"=== Run: {run_dir.name} (config: {config.name}) ===")

    if "prep" in stages:
        prep_l1(config, run_dir, project_root, items_file=args.items_file)
    if "generate" in stages:
        await generate_l1(config, run_dir, project_root)
    if "validate" in stages:
        await validate_l1(config, run_dir, project_root)

    # --- Variante A: Aggregate (softmax over validation scores) ---
    if "aggregate" in stages:
        aggregate_l1(config, run_dir, project_root)
        # Use aggregate distributions for metrics + analysis
        shutil.copy2(run_dir / "distributions_aggregate.jsonl", run_dir / "distributions.jsonl")
        write_metrics_l1(config, run_dir)
        analysis_dir_a = run_dir / "analysis_aggregate"
        analysis_dir_a.mkdir(parents=True, exist_ok=True)
        run_analyze(config, run_dir, project_root)
        # Move analysis artifacts to analysis_aggregate/
        for f in (run_dir / "analysis").iterdir():
            shutil.copy2(f, analysis_dir_a / f.name)
        shutil.copy2(run_dir / "metrics.json", run_dir / "metrics_aggregate.json")
        print(f"[Variante A] Aggregate results in {analysis_dir_a}")

    # --- Variante B: Compare (best explanation ranked comparatively) ---
    if "compare" in stages:
        await compare_l1(config, run_dir, project_root)
        # distributions.jsonl is already written by compare_l1
        write_metrics_l1(config, run_dir)
        analysis_dir_b = run_dir / "analysis_compare"
        analysis_dir_b.mkdir(parents=True, exist_ok=True)
        run_analyze(config, run_dir, project_root)
        for f in (run_dir / "analysis").iterdir():
            shutil.copy2(f, analysis_dir_b / f.name)
        shutil.copy2(run_dir / "metrics.json", run_dir / "metrics_compare.json")
        print(f"[Variante B] Compare results in {analysis_dir_b}")

    print(f"\n=== Done. Results in {run_dir} ===")


if __name__ == "__main__":
    asyncio.run(main())
