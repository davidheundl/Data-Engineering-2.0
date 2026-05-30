"""Stage 5: Analyze — quantitative + qualitative analysis. No LLM calls.

Produces these artifacts in results/{run_id}/analysis/:
  1. kld_curve.png       — KLD(crowd || LLM) vs tau, overall + per validator-family
  2. validation_overlap.png — precision/recall of LLM-validated senses against
                              crowd-majority pseudo-GT, swept over tau
  3. cross_family_signal.json + scatter plot — validator_std signal
  4. commission_error_candidates.csv — ranked (item, sense) pairs
  5. per_genre_breakdown.json — all metrics by genre
  6. worked_example.json — one item traced end-to-end
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .aggregate import _kld, read_distributions
from .config import Config
from .generate import read_generations
from .prep import read_items
from .schemas import (
    DistributionRecord,
    GenerationRecord,
    PrepItem,
    ValidationRecord,
)
from .validate import read_validations


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _kld_curve(
    items: list[PrepItem],
    dists: list[DistributionRecord],
    vals: list[ValidationRecord],
    config: Config,
    out_path: Path,
) -> None:
    """KLD(crowd || LLM) vs tau. One curve per validator family + aggregate.

    To produce a per-validator curve, we rebuild the LLM distribution using
    only that validator's scores. This shows whether a single validator
    family is more or less aligned with the crowd.
    """
    items_by_id = {it.item_id: it for it in items}

    # Group validator scores by (item, sense, validator)
    scores_by: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for v in vals:
        scores_by[(v.item_id, v.candidate_sense, v.validator_model)].append(v.validity_score)

    taus = config.pipeline.tau_values()

    def _max_validity(item_id: str, sense: str, validator: str | None) -> float:
        # Aggregate max mean-validity across explanations for this validator (or all)
        if validator is None:
            # average across all validators per explanation; take max
            return _max_validity_all(item_id, sense, scores_by)
        # only this validator: each explanation contributes its score
        scores: list[float] = []
        for (iid, s, vname), sc in scores_by.items():
            if iid == item_id and s == sense and vname == validator:
                scores.extend(sc)
        return max(scores) if scores else 0.0

    # ----- Aggregate curve (uses precomputed distributions in dists) -----
    aggregate_kld: list[float] = []
    for tau in taus:
        key = f"{tau:.2f}"
        klds: list[float] = []
        for rec in dists:
            llm = rec.llm_label_distribution_per_tau.get(key, {})
            if not llm:
                continue
            klds.append(_kld(items_by_id[rec.item_id].crowd_sense_distribution, llm))
        aggregate_kld.append(mean(klds) if klds else float("nan"))

    # ----- Per-validator curves -----
    # NOTE: Per-validator curves use equal-mass distribution (not softmax)
    # for simplicity, since per-validator mean_validity is not precomputed.
    # The aggregate curve above uses softmax via distributions.jsonl.
    validator_curves: dict[str, list[float]] = {}
    for validator in config.models.validators:
        curve: list[float] = []
        for tau in taus:
            klds: list[float] = []
            for rec in dists:
                # Build distribution using only this validator's max_validity
                validated = []
                for sense in rec.per_sense_stats:
                    mv = _max_validity(rec.item_id, sense, validator)
                    if mv >= tau:
                        validated.append(sense)
                if not validated:
                    continue
                llm = {s: 1.0 / len(validated) for s in validated}
                klds.append(_kld(items_by_id[rec.item_id].crowd_sense_distribution, llm))
            curve.append(mean(klds) if klds else float("nan"))
        validator_curves[validator] = curve

    plt.figure(figsize=(8, 5))
    plt.plot(taus, aggregate_kld, label="aggregate (all validators)", linewidth=2.5, color="black")
    palette = sns.color_palette("tab10", n_colors=len(validator_curves))
    for (validator, curve), color in zip(validator_curves.items(), palette):
        plt.plot(taus, curve, label=validator, linewidth=1.5, color=color, marker="o")
    plt.xlabel("tau (validity threshold)")
    plt.ylabel("mean KLD(crowd || LLM)")
    plt.title("Divergence between crowd and LLM-derived label distributions")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _max_validity_all(
    item_id: str,
    sense: str,
    scores_by: dict[tuple[str, str, str], list[float]],
) -> float:
    """Mean-across-validators-then-max-across-explanations.

    For each explanation (identified by all validator scores tied to it),
    average across validators. Then take max across explanations.

    Since we don't have explanation indices in scores_by, we approximate by
    averaging *all* scores per (item, sense) — used only inside KLD curve
    aggregate fallback (not the same as per-explanation max). For correctness
    we use the precomputed distributions instead in the main aggregate.
    """
    scores: list[float] = []
    for (iid, s, _v), sc in scores_by.items():
        if iid == item_id and s == sense:
            scores.extend(sc)
    return max(scores) if scores else 0.0


def _validation_overlap(
    items: list[PrepItem],
    dists: list[DistributionRecord],
    config: Config,
    out_path: Path,
) -> None:
    """Precision/recall of LLM-validated senses vs crowd-majority pseudo-GT.

    Pseudo-GT: senses with crowd_proportion > 0 (or top-1 if strict).
    For each tau: validated senses = {s : max_validity(s) >= tau}.
    Precision = |validated ∩ crowd_present| / |validated|
    Recall    = |validated ∩ crowd_present| / |crowd_present|
    """
    items_by_id = {it.item_id: it for it in items}
    taus = config.pipeline.tau_values()

    precisions: list[float] = []
    recalls: list[float] = []
    for tau in taus:
        key = f"{tau:.2f}"
        p_vals: list[float] = []
        r_vals: list[float] = []
        for rec in dists:
            validated = set(rec.llm_label_distribution_per_tau.get(key, {}).keys())
            crowd_present = set(items_by_id[rec.item_id].crowd_sense_distribution.keys())
            tp = len(validated & crowd_present)
            if validated:
                p_vals.append(tp / len(validated))
            if crowd_present:
                r_vals.append(tp / len(crowd_present))
        precisions.append(mean(p_vals) if p_vals else float("nan"))
        recalls.append(mean(r_vals) if r_vals else float("nan"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(taus, precisions, marker="o", label="precision", color="tab:blue")
    ax[0].plot(taus, recalls, marker="s", label="recall", color="tab:orange")
    ax[0].set_xlabel("tau")
    ax[0].set_ylabel("score")
    ax[0].set_title("Precision / Recall vs tau")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(recalls, precisions, marker="o")
    for i, tau in enumerate(taus):
        ax[1].annotate(f"{tau:.1f}", (recalls[i], precisions[i]), fontsize=7)
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_title("PR curve (annotated with tau)")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _cross_family_signal(
    items: list[PrepItem],
    dists: list[DistributionRecord],
    analysis_dir: Path,
) -> None:
    """validator_std vs crowd_agreement_score and vs ref label presence.

    Saves cross_family_signal.json (correlations) + scatter plot.
    """
    items_by_id = {it.item_id: it for it in items}

    points: list[dict[str, Any]] = []
    for rec in dists:
        item = items_by_id[rec.item_id]
        has_ref = item.wikipedia_reference_labels is not None
        for sense, st in rec.per_sense_stats.items():
            points.append(
                {
                    "item_id": rec.item_id,
                    "genre": item.genre,
                    "sense": sense,
                    "validator_std": st.validator_std,
                    "crowd_agreement_score": item.crowd_agreement_score,
                    "has_ref": has_ref,
                    "commission_score": st.commission_score,
                }
            )

    # Correlations
    def _pearson(xs: list[float], ys: list[float]) -> float:
        if len(xs) < 2:
            return float("nan")
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if dx == 0 or dy == 0:
            return float("nan")
        return num / (dx * dy)

    stds = [p["validator_std"] for p in points]
    agreements = [p["crowd_agreement_score"] for p in points]
    ref_flags = [1.0 if p["has_ref"] else 0.0 for p in points]
    payload = {
        "n_points": len(points),
        "pearson_std_vs_crowd_agreement": _pearson(stds, agreements),
        "pearson_std_vs_has_reference_label": _pearson(stds, ref_flags),
        "mean_std_with_ref": mean([p["validator_std"] for p in points if p["has_ref"]])
        if any(p["has_ref"] for p in points)
        else None,
        "mean_std_without_ref": mean([p["validator_std"] for p in points if not p["has_ref"]])
        if any(not p["has_ref"] for p in points)
        else None,
    }
    with (analysis_dir / "cross_family_signal.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # Scatter: validator_std vs crowd_agreement_score, colored by genre
    fig, ax = plt.subplots(figsize=(7, 5))
    for genre, color in zip(["Europarl", "Lit", "Wiki"], ["tab:blue", "tab:orange", "tab:green"]):
        xs = [p["crowd_agreement_score"] for p in points if p["genre"] == genre]
        ys = [p["validator_std"] for p in points if p["genre"] == genre]
        ax.scatter(xs, ys, label=genre, alpha=0.5, color=color)
    ax.set_xlabel("crowd_agreement_score")
    ax.set_ylabel("validator_std")
    ax.set_title("Validator disagreement vs crowd agreement")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_dir / "cross_family_signal_scatter.png", dpi=120)
    plt.close()


def _commission_candidates(
    items: list[PrepItem],
    dists: list[DistributionRecord],
    gens: list[GenerationRecord],
    vals: list[ValidationRecord],
    out_path: Path,
) -> None:
    """Ranked CSV of (item_id, sense) by commission_score desc."""
    items_by_id = {it.item_id: it for it in items}
    expls_by: dict[tuple[str, str], list[str]] = defaultdict(list)
    for g in gens:
        if g.abstained:
            continue
        for e in g.explanations:
            expls_by[(g.item_id, g.candidate_sense)].append(f"[{g.generator_model}] {e}")
    scores_by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for v in vals:
        scores_by[(v.item_id, v.candidate_sense)].append(v.validity_score)

    rows: list[dict[str, Any]] = []
    for rec in dists:
        item = items_by_id[rec.item_id]
        for sense, st in rec.per_sense_stats.items():
            rows.append(
                {
                    "rank": 0,
                    "item_id": rec.item_id,
                    "genre": item.genre,
                    "candidate_sense": sense,
                    "commission_score": round(st.commission_score, 4),
                    "max_validity": round(st.max_validity, 4),
                    "mean_validity": round(st.mean_validity, 4),
                    "validator_std": round(st.validator_std, 4),
                    "n_explanations": st.n_explanations,
                    "n_generators_abstained": st.n_generators_abstained,
                    "crowd_proportion": round(
                        item.crowd_sense_distribution.get(sense, 0.0), 4
                    ),
                    "crowd_agreement": round(item.crowd_agreement_score, 4),
                    "arg1": item.arg1,
                    "arg2": item.arg2,
                    "explanations": " || ".join(expls_by.get((rec.item_id, sense), [])),
                    "all_validity_scores": ";".join(
                        f"{s:.2f}" for s in scores_by.get((rec.item_id, sense), [])
                    ),
                    "crowd_distribution": json.dumps(item.crowd_sense_distribution),
                }
            )

    rows.sort(key=lambda r: r["commission_score"], reverse=True)
    rows = rows[:1000]
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
        r["flagged_top10"] = i <= 10

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _per_genre_breakdown(
    items: list[PrepItem],
    dists: list[DistributionRecord],
    config: Config,
    out_path: Path,
) -> None:
    items_by_id = {it.item_id: it for it in items}
    taus = config.pipeline.tau_values()

    out: dict[str, Any] = {}
    for genre in ["Europarl", "Lit", "Wiki"]:
        item_ids = {it.item_id for it in items if it.genre == genre}
        if not item_ids:
            continue
        genre_dists = [d for d in dists if d.item_id in item_ids]

        # KLD per tau (genre-only)
        kld_per_tau: dict[str, float] = {}
        for tau in taus:
            key = f"{tau:.2f}"
            klds: list[float] = []
            for rec in genre_dists:
                llm = rec.llm_label_distribution_per_tau.get(key, {})
                if not llm:
                    continue
                klds.append(_kld(items_by_id[rec.item_id].crowd_sense_distribution, llm))
            kld_per_tau[key] = mean(klds) if klds else None

        # mean commission_score (max per item)
        max_commissions: list[float] = []
        for rec in genre_dists:
            if rec.per_sense_stats:
                max_commissions.append(max(st.commission_score for st in rec.per_sense_stats.values()))
        out[genre] = {
            "n_items": len(item_ids),
            "kld_vs_crowd_per_tau": kld_per_tau,
            "mean_max_commission_score": mean(max_commissions) if max_commissions else None,
            "std_max_commission_score": pstdev(max_commissions) if len(max_commissions) > 1 else 0.0,
        }

    with out_path.open("w") as f:
        json.dump(out, f, indent=2)


def _worked_example(
    items: list[PrepItem],
    gens: list[GenerationRecord],
    vals: list[ValidationRecord],
    dists: list[DistributionRecord],
    out_path: Path,
) -> None:
    """Pick one item with at least one commission_score > 0.3 and trace it."""
    if not items:
        return

    # Choose the item with the highest single commission_score for narrative value
    candidate_item_id: str | None = None
    best_score = -1.0
    for rec in dists:
        for sense, st in rec.per_sense_stats.items():
            if st.commission_score > best_score:
                best_score = st.commission_score
                candidate_item_id = rec.item_id
    if candidate_item_id is None:
        candidate_item_id = items[0].item_id

    item = next(it for it in items if it.item_id == candidate_item_id)
    item_gens = [g for g in gens if g.item_id == candidate_item_id]
    item_vals = [v for v in vals if v.item_id == candidate_item_id]
    item_dist = next((d for d in dists if d.item_id == candidate_item_id), None)

    payload = {
        "stage1_item": json.loads(item.model_dump_json()),
        "stage2_generations": [json.loads(g.model_dump_json()) for g in item_gens],
        "stage3_validations": [json.loads(v.model_dump_json()) for v in item_vals],
        "stage4_distribution": json.loads(item_dist.model_dump_json()) if item_dist else None,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_analyze(config: Config, run_dir: Path, project_root: Path) -> Path:
    items = read_items(run_dir / "items.jsonl")
    gens = read_generations(run_dir / "generations.jsonl")
    vals = read_validations(run_dir / "validations.jsonl")
    dists = read_distributions(run_dir / "distributions.jsonl")

    analysis_dir = _ensure_dir(run_dir / "analysis")

    _kld_curve(items, dists, vals, config, analysis_dir / "kld_curve.png")
    _validation_overlap(items, dists, config, analysis_dir / "validation_overlap.png")
    _cross_family_signal(items, dists, analysis_dir)
    _commission_candidates(
        items, dists, gens, vals, analysis_dir / "commission_error_candidates.csv"
    )
    _per_genre_breakdown(items, dists, config, analysis_dir / "per_genre_breakdown.json")
    _worked_example(items, gens, vals, dists, analysis_dir / "worked_example.json")

    print(f"[Stage 5] Wrote analysis artifacts to {analysis_dir}")
    return analysis_dir
