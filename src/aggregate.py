"""Stage 4: Aggregate — per-item / per-sense statistics, threshold sweep,
LLM label distributions, and global metrics. No LLM calls.

For each (item, candidate_sense):
  1. Collect non-abstained explanations across all generators.
  2. For each explanation, compute the mean validity across validators.
  3. Compute stats: max_validity, mean_validity, n_explanations,
     n_generators_abstained, validator_std (avg stddev of validator scores
     per explanation), commission_score = (1 - max_validity) * (1 - validator_std).
  4. Sweep tau in [0.1, 0.9, 0.1]: build the LLM label distribution as
     equal mass on validated senses (max_validity >= tau), normalized.

Writes distributions.jsonl + metrics.json.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from .config import Config
from .generate import read_generations
from .prep import read_items
from .schemas import (
    DistributionRecord,
    GenerationRecord,
    PrepItem,
    RunMetrics,
    SenseStats,
    ValidationRecord,
)
from .validate import read_validations


def _kld(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    """KL(p || q) over the union of keys, with epsilon smoothing."""
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pv = p.get(k, 0.0)
        qv = q.get(k, 0.0)
        if pv <= 0.0:
            continue
        out += pv * math.log((pv + eps) / (qv + eps))
    return out


def _validator_std_per_explanation(scores: list[float]) -> float:
    """Population stddev of validator scores. Returns 0.0 if fewer than 2 scores."""
    if len(scores) < 2:
        return 0.0
    return pstdev(scores)


def _build_llm_distribution(
    per_sense_stats: dict[str, SenseStats], tau: float
) -> dict[str, float]:
    """Equal mass on senses with max_validity >= tau, normalized.

    If no senses are validated, return an empty dict (signals no prediction).
    """
    validated = [s for s, st in per_sense_stats.items() if st.max_validity >= tau]
    if not validated:
        return {}
    p = 1.0 / len(validated)
    return {s: p for s in validated}


def _aggregate_one_item(
    item: PrepItem,
    item_gens: list[GenerationRecord],
    item_vals: list[ValidationRecord],
    config: Config,
) -> tuple[DistributionRecord, dict[str, SenseStats]]:
    # Group validations by (sense, explanation_text) -> list of validator scores
    vals_by_sense_expl: dict[tuple[str, str], list[float]] = defaultdict(list)
    for v in item_vals:
        vals_by_sense_expl[(v.candidate_sense, v.explanation_text)].append(v.validity_score)

    # Group generations by sense -> all explanations + abstention counts
    expls_by_sense: dict[str, list[str]] = defaultdict(list)
    abstain_count: dict[str, int] = defaultdict(int)
    for g in item_gens:
        if g.abstained or not g.explanations:
            abstain_count[g.candidate_sense] += 1
        for expl in g.explanations:
            expls_by_sense[g.candidate_sense].append(expl)

    per_sense_stats: dict[str, SenseStats] = {}
    for sense in item.candidate_senses:
        explanations = expls_by_sense.get(sense, [])
        # mean validity per explanation (avg across validators)
        per_expl_means: list[float] = []
        per_expl_stds: list[float] = []
        for expl in explanations:
            scores = vals_by_sense_expl.get((sense, expl), [])
            if not scores:
                continue
            per_expl_means.append(mean(scores))
            per_expl_stds.append(_validator_std_per_explanation(scores))

        if per_expl_means:
            max_v = max(per_expl_means)
            mean_v = mean(per_expl_means)
            v_std = mean(per_expl_stds) if per_expl_stds else 0.0
        else:
            max_v = 0.0
            mean_v = 0.0
            v_std = 0.0

        # Clamp std to [0,1] for commission_score formula
        v_std_clamped = min(1.0, v_std)
        commission = (1.0 - max_v) * (1.0 - v_std_clamped)
        commission = max(0.0, min(1.0, commission))

        per_sense_stats[sense] = SenseStats(
            max_validity=max_v,
            mean_validity=mean_v,
            n_explanations=len(per_expl_means),
            n_generators_abstained=abstain_count.get(sense, 0),
            validator_std=v_std,
            commission_score=commission,
        )

    # tau sweep
    llm_dist_per_tau: dict[str, dict[str, float]] = {}
    for tau in config.pipeline.tau_values():
        key = f"{tau:.2f}"
        llm_dist_per_tau[key] = _build_llm_distribution(per_sense_stats, tau)

    record = DistributionRecord(
        item_id=item.item_id,
        candidate_senses=item.candidate_senses,
        per_sense_stats=per_sense_stats,
        llm_label_distribution_per_tau=llm_dist_per_tau,
    )
    return record, per_sense_stats


def run_aggregate(config: Config, run_dir: Path, project_root: Path) -> tuple[Path, Path]:
    """Run Stage 4. Returns (distributions.jsonl, metrics.json) paths."""
    items = read_items(run_dir / "items.jsonl")
    gens = read_generations(run_dir / "generations.jsonl")
    vals = read_validations(run_dir / "validations.jsonl")

    gens_by_item: dict[str, list[GenerationRecord]] = defaultdict(list)
    for g in gens:
        gens_by_item[g.item_id].append(g)
    vals_by_item: dict[str, list[ValidationRecord]] = defaultdict(list)
    for v in vals:
        vals_by_item[v.item_id].append(v)

    items_by_id = {it.item_id: it for it in items}
    item_genre: dict[str, str] = {it.item_id: it.genre for it in items}

    dist_path = run_dir / "distributions.jsonl"
    records: list[DistributionRecord] = []
    with dist_path.open("w") as f:
        for item in items:
            rec, _ = _aggregate_one_item(
                item, gens_by_item.get(item.item_id, []), vals_by_item.get(item.item_id, []), config
            )
            f.write(rec.model_dump_json() + "\n")
            records.append(rec)

    # ----- Global metrics -----
    total_gen_calls = len(gens)
    total_val_calls = len(vals)
    total_cost = sum(g.cost_usd for g in gens) + sum(v.cost_usd for v in vals)
    mean_cost_per_item = total_cost / max(1, len(items))

    # Abstention rate per generator model
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

    # Avg commission score by genre — average max commission_score across senses per item
    genre_scores: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if not rec.per_sense_stats:
            continue
        max_commission = max(st.commission_score for st in rec.per_sense_stats.values())
        genre_scores[item_genre[rec.item_id]].append(max_commission)
    avg_commission_by_genre = {
        g: (mean(scores) if scores else 0.0) for g, scores in genre_scores.items()
    }

    # KLD vs crowd per tau (mean across items where LLM dist is non-empty)
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

    print(f"[Stage 4] Wrote {dist_path} and {metrics_path}")
    print(f"  total cost ${total_cost:.4f}, abstention rates: {abstention_rate}")
    return dist_path, metrics_path


def read_distributions(distributions_jsonl: Path) -> list[DistributionRecord]:
    records: list[DistributionRecord] = []
    with distributions_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(DistributionRecord(**json.loads(line)))
    return records
