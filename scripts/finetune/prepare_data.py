#!/usr/bin/env python3
"""Build train/test datasets for the fine-tuning experiment.

Produces four files in --out-dir:
  - train_raw_crowd.jsonl       : Crowd L1 distribution as target
  - train_raw_evade.jsonl       : Variante-B LLM distribution as target
  - train_quadrant_curated.jsonl: Crowd distribution with senses marked as
                                  `llm_overconfident` or `high_risk` zeroed
                                  out, then renormalized.
  - test.jsonl                  : Held-out items with both crowd and
                                  Wikipedia editor-gold distributions
                                  attached for evaluation.

Test items are stratified by Wikipedia-gold availability so the test set
gets the maximum gold signal possible.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SENSES = ["temporal", "contingency", "comparison", "expansion"]

PDTB_REFLABEL_TO_L1 = {
    "synchronous": "temporal", "asynchronous": "temporal",
    "precedence": "temporal", "succession": "temporal",
    "reason": "contingency", "result": "contingency",
    "arg1-as-cond": "contingency", "arg2-as-cond": "contingency",
    "arg1-as-negcond": "contingency", "arg2-as-negcond": "contingency",
    "arg1-as-goal": "contingency", "arg2-as-goal": "contingency",
    "arg2-as-purpose": "contingency",
    "contrast": "comparison", "similarity": "comparison", "concession": "comparison",
    "arg1-as-denier": "comparison", "arg2-as-denier": "comparison",
    "conjunction": "expansion", "disjunction": "expansion", "equivalence": "expansion",
    "arg1-as-detail": "expansion", "arg2-as-detail": "expansion",
    "arg1-as-instance": "expansion", "arg2-as-instance": "expansion",
    "arg1-as-subst": "expansion", "arg2-as-subst": "expansion",
    "arg1-as-excpt": "expansion", "arg2-as-excpt": "expansion",
    "arg1-as-manner": "expansion", "arg2-as-manner": "expansion",
}


def crowd_l1(item: dict) -> dict[str, float]:
    d = item["crowd_sense_distribution"]
    filtered = {s: d.get(s, 0.0) for s in SENSES}
    total = sum(filtered.values())
    if total <= 0:
        return {s: 0.25 for s in SENSES}
    return {s: v / total for s, v in filtered.items()}


def normalize(d: dict[str, float], fallback: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    if total <= 0:
        return fallback
    return {s: d[s] / total for s in SENSES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Wiki-extension run directory")
    parser.add_argument("--wide-csv", default="DiscoGeM 1.0_items/DiscoGeM1.0.wide.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load items.jsonl
    items: dict[str, dict] = {}
    for line in (run_dir / "items.jsonl").open():
        r = json.loads(line)
        items[r["item_id"]] = r
    print(f"Loaded {len(items)} items")

    # 2. Load distributions.jsonl (Variante B). Take tau=0.30.
    evade_dist: dict[str, dict[str, float]] = {}
    for line in (run_dir / "distributions.jsonl").open():
        r = json.loads(line)
        per_tau = r.get("llm_label_distribution_per_tau", {})
        raw = per_tau.get("0.30") or per_tau.get("0.30000000000000004") or {}
        if not raw:
            continue
        dist = {s: raw.get(s, 0.0) for s in SENSES}
        crowd_fb = crowd_l1(items[r["item_id"]]) if r["item_id"] in items else {s: 0.25 for s in SENSES}
        evade_dist[r["item_id"]] = normalize(dist, crowd_fb)
    print(f"Loaded {len(evade_dist)} EVADE distributions")

    # 3. Load risk_quadrants.csv (per item -> {sense: quadrant_B})
    quadrants: dict[str, dict[str, str]] = defaultdict(dict)
    q_path = run_dir / "analysis_detailed" / "risk_quadrants.csv"
    for r in csv.DictReader(q_path.open()):
        quadrants[r["item_id"]][r["sense"]] = r.get("quadrant_B") or r.get("quadrant_A", "")
    print(f"Loaded quadrants for {len(quadrants)} items")

    # 4. Load Wikipedia editor gold (PDTB-3 reflabel -> L1 set)
    wiki_gold: dict[str, set[str]] = {}
    with open(args.wide_csv, newline="") as f:
        for r in csv.DictReader(f):
            ref = (r.get("reflabel") or "").strip()
            if not ref or ref.upper() == "NA":
                continue
            l1_set = set()
            for tok in ref.split(";"):
                tok = tok.strip()
                if not tok or tok.lower() == "norel":
                    continue
                l1 = PDTB_REFLABEL_TO_L1.get(tok)
                if l1:
                    l1_set.add(l1)
            if l1_set:
                wiki_gold[r["itemid"]] = l1_set
    print(f"Loaded {len(wiki_gold)} wiki gold labels (in corpus)")

    # 5. Curated distribution = crowd, with `llm_overconfident` and
    #    `high_risk` senses zeroed out, then renormalized.
    def curated(item: dict) -> dict[str, float]:
        crowd = crowd_l1(item)
        out = dict(crowd)
        quads = quadrants.get(item["item_id"], {})
        for s in SENSES:
            q = quads.get(s)
            if q in ("llm_overconfident", "high_risk"):
                out[s] = 0.0
        return normalize(out, fallback=crowd)

    # 6. Train/test split, stratified by gold-availability.
    rng = random.Random(args.seed)
    all_ids = sorted(items.keys())
    with_gold = [i for i in all_ids if i in wiki_gold]
    without_gold = [i for i in all_ids if i not in wiki_gold]
    rng.shuffle(with_gold)
    rng.shuffle(without_gold)
    n_test = int(round(len(all_ids) * args.test_frac))
    # Test items: take as many gold items as we can to maximize eval signal.
    n_test_gold = min(n_test, len(with_gold))
    test_ids = set(with_gold[:n_test_gold] + without_gold[:max(0, n_test - n_test_gold)])
    train_ids = set(all_ids) - test_ids
    print(f"Split: {len(train_ids)} train, {len(test_ids)} test "
          f"(test items with gold: {sum(1 for i in test_ids if i in wiki_gold)})")

    # 7. Write three train variants
    counts = {"raw_crowd": 0, "raw_evade": 0, "quadrant_curated": 0}
    for variant in ["raw_crowd", "raw_evade", "quadrant_curated"]:
        path = out_dir / f"train_{variant}.jsonl"
        with path.open("w") as f:
            for item_id in sorted(train_ids):
                item = items[item_id]
                if variant == "raw_crowd":
                    label = crowd_l1(item)
                elif variant == "raw_evade":
                    label = evade_dist.get(item_id)
                    if label is None:
                        continue
                else:
                    label = curated(item)
                f.write(json.dumps({
                    "item_id": item_id,
                    "arg1": item["arg1"],
                    "arg2": item["arg2"],
                    "label_distribution": label,
                }) + "\n")
                counts[variant] += 1
        print(f"  Wrote {path} ({counts[variant]} examples)")

    # 8. Write test set with crowd + gold
    path = out_dir / "test.jsonl"
    n_with_gold = 0
    with path.open("w") as f:
        for item_id in sorted(test_ids):
            item = items[item_id]
            crowd = crowd_l1(item)
            gold = sorted(wiki_gold.get(item_id, set()))
            gold_dist = None
            if gold:
                n_with_gold += 1
                gold_dist = {s: (1.0 / len(gold) if s in gold else 0.0) for s in SENSES}
            f.write(json.dumps({
                "item_id": item_id,
                "arg1": item["arg1"],
                "arg2": item["arg2"],
                "crowd_distribution": crowd,
                "gold_senses": gold,
                "gold_distribution": gold_dist,
            }) + "\n")
    print(f"  Wrote {path} ({len(test_ids)} items, {n_with_gold} with wiki gold)")


if __name__ == "__main__":
    main()
