"""Stage 1: Prep — turn raw DiscoGeM into items.jsonl.

Reads:
  - DiscoGeM 1.0_items/DiscoGeM1.0.wide.csv     (item metadata)
  - DiscoGeM 1.0_labels/DiscoGeMcorpus_fulldataset.csv  (per-worker annotations)

For each item:
  1. Collect per-annotator Level-2 senses (lev2_conn2), dropping "NA" entries.
  2. Compute the normalized crowd sense distribution.
  3. Candidate senses = all senses with >= 1 vote.
  4. Stratify by (genre, agreement_bin).
  5. Sample 50 items: 17 Europarl / 17 Lit / 16 Wiki, half high / half low
     agreement (high = crowd_agreement_score >= 0.5).

Writes one PrepItem per line to {results_dir}/{run_id}/items.jsonl.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import Config
from .schemas import PrepItem

GENRE_MAP = {"novel": "Lit", "europarl": "Europarl", "wikipedia": "Wiki"}
AGREEMENT_HIGH_THRESHOLD = 0.5


def _split_high_low(quota: int) -> tuple[int, int]:
    """Half high / half low. If quota is odd, give the extra to 'high'."""
    high = math.ceil(quota / 2)
    low = quota - high
    return high, low


def _build_crowd_distribution(senses: list[str]) -> dict[str, float]:
    """Normalize a list of per-annotator senses into a probability distribution."""
    counts = Counter(senses)
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()}


def _parse_reflabel(raw: object) -> list[str] | None:
    """Parse wiki/PDTB reference labels. Returns None if NA, else list of senses."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped or stripped.upper() == "NA":
        return None
    return [s.strip() for s in stripped.split(";") if s.strip()]


def load_items(config: Config, *, project_root: Path) -> pd.DataFrame:
    """Load wide + full datasets and produce a per-item DataFrame.

    Each row has: item_id, genre, arg1, arg2, arg1_singlesentence,
    arg2_singlesentence, reflabel, split, annotator_senses (list[str]).
    """
    wide_path = project_root / config.data.wide_csv
    full_path = project_root / config.data.full_csv

    wide = pd.read_csv(wide_path, low_memory=False)
    full = pd.read_csv(full_path, low_memory=False)

    # Filter to the three genres in scope
    wide = wide[wide["genre"].isin(GENRE_MAP.keys())].copy()
    full = full[full["genre"].isin(GENRE_MAP.keys())].copy()

    # Drop annotations with NA Level-2 sense
    full = full[full["lev2_conn2"].notna()].copy()
    full = full[full["lev2_conn2"].astype(str).str.upper() != "NA"].copy()

    # Some lev2_conn2 entries are multi-valued (comma-separated). For Level-2
    # we keep only single-sense entries to avoid double-counting; multi-valued
    # entries are rare and ambiguous in the corpus.
    full = full[~full["lev2_conn2"].astype(str).str.contains(",")].copy()

    # Group per-item annotator senses
    sense_lists = (
        full.groupby("itemid")["lev2_conn2"].apply(list).rename("annotator_senses")
    )

    df = wide.merge(sense_lists, left_on="itemid", right_index=True, how="inner")

    # Map raw genre -> three-genre label
    df["genre_mapped"] = df["genre"].map(GENRE_MAP)
    return df


def _build_prep_item(row: pd.Series) -> PrepItem:
    senses = list(row["annotator_senses"])
    distribution = _build_crowd_distribution(senses)
    # Drop "norel" (no-relation) from candidate_senses: it is not a discourse
    # sense we want to generate/validate explanations for. Keep it in the
    # crowd_sense_distribution so the KLD comparison against crowd is honest.
    candidate_senses = sorted(s for s in distribution.keys() if s != "norel")
    majority = max(distribution.items(), key=lambda kv: kv[1])[0]
    agreement = max(distribution.values())
    bin_ = "high" if agreement >= AGREEMENT_HIGH_THRESHOLD else "low"
    return PrepItem(
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
        wikipedia_reference_labels=_parse_reflabel(row.get("reflabel")),
        crowd_agreement_score=agreement,
        stratification_bin=bin_,
        split=str(row["split"]) if "split" in row and pd.notna(row.get("split")) else None,
    )


def stratified_sample(
    candidates: list[PrepItem], config: Config
) -> list[PrepItem]:
    """Stratified sample by (genre, agreement_bin) using config.sampling.seed."""
    import random

    rng = random.Random(config.sampling.seed)
    selected: list[PrepItem] = []

    for genre, quota in config.sampling.genre_split.items():
        pool = [c for c in candidates if c.genre == genre]
        # If the quota covers the entire pool, take all items (skip stratification).
        if quota >= len(pool):
            pool.sort(key=lambda c: c.item_id)
            selected.extend(pool)
            continue
        high_quota, low_quota = _split_high_low(quota)
        high_pool = [c for c in pool if c.stratification_bin == "high"]
        low_pool = [c for c in pool if c.stratification_bin == "low"]

        if len(high_pool) < high_quota:
            raise ValueError(
                f"Not enough 'high' agreement items for {genre}: "
                f"need {high_quota}, have {len(high_pool)}"
            )
        if len(low_pool) < low_quota:
            raise ValueError(
                f"Not enough 'low' agreement items for {genre}: "
                f"need {low_quota}, have {len(low_pool)}"
            )
        # Sort for determinism, then random.sample with seeded rng
        high_pool.sort(key=lambda c: c.item_id)
        low_pool.sort(key=lambda c: c.item_id)
        selected.extend(rng.sample(high_pool, high_quota))
        selected.extend(rng.sample(low_pool, low_quota))

    return selected


def run_prep(config: Config, run_dir: Path, project_root: Path) -> Path:
    """Run Stage 1. Returns the path to items.jsonl."""
    df = load_items(config, project_root=project_root)

    # Require at least 2 valid annotations to compute a meaningful distribution
    candidates: list[PrepItem] = []
    for _, row in df.iterrows():
        if len(row["annotator_senses"]) < 2:
            continue
        candidates.append(_build_prep_item(row))

    sampled = stratified_sample(candidates, config)

    out_path = run_dir / "items.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in sampled:
            f.write(item.model_dump_json() + "\n")

    print(
        f"[Stage 1] Wrote {len(sampled)} items to {out_path} "
        f"(from pool of {len(candidates)} candidates)"
    )
    # Per-genre / agreement summary
    counts: dict[tuple[str, str], int] = {}
    for it in sampled:
        counts[(it.genre, it.stratification_bin)] = counts.get((it.genre, it.stratification_bin), 0) + 1
    for (g, b), n in sorted(counts.items()):
        print(f"  {g:<10s} {b:<5s} -> {n}")
    return out_path


def read_items(items_jsonl: Path) -> list[PrepItem]:
    """Read items.jsonl back, validating against PrepItem schema."""
    items: list[PrepItem] = []
    with items_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(PrepItem(**json.loads(line)))
    return items
