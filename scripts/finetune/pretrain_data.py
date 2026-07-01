#!/usr/bin/env python3
"""Build pre-training dataset from DiscoGeM.

Extracts ~5800 items NOT in the Wiki-extension run and writes them with
their crowd-derived L1 distributions to pretrain.jsonl. No LLM calls -
this uses only the original DiscoGeM crowd annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SENSES = ["temporal", "contingency", "comparison", "expansion"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-csv", default="DiscoGeM 1.0_items/DiscoGeM1.0.wide.csv")
    parser.add_argument("--full-csv", default="DiscoGeM 1.0_labels/DiscoGeMcorpus_fulldataset.csv")
    parser.add_argument("--exclude-run-dir",
                        default="results/20260625T062938Z_wiki_extension_bcb6af9",
                        help="Run directory whose items.jsonl will be excluded.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-annotators", type=int, default=3,
                        help="Drop items with fewer than this many valid L1 annotations.")
    args = parser.parse_args()

    # Items to hold out (= our fine-tune train+test set)
    excluded = set()
    for line in Path(args.exclude_run_dir, "items.jsonl").open():
        excluded.add(json.loads(line)["item_id"])
    print(f"Excluding {len(excluded)} items from {args.exclude_run_dir}")

    # Read wide CSV: arg1, arg2, genre per item
    items: dict[str, dict] = {}
    with open(args.wide_csv, newline="") as f:
        for r in csv.DictReader(f):
            iid = r.get("itemid")
            if not iid:
                continue
            arg1 = (r.get("arg1") or "").strip()
            arg2 = (r.get("arg2") or "").strip()
            if not arg1 or not arg2:
                continue
            items[iid] = {
                "item_id": iid,
                "arg1": arg1,
                "arg2": arg2,
                "genre": (r.get("genre") or "").strip(),
                "annotator_senses": [],
            }
    print(f"Loaded {len(items)} items with arg1+arg2 from wide CSV")

    # Read per-annotator L1 labels from full CSV
    n_rows = 0
    with open(args.full_csv, newline="") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            iid = r.get("itemid")
            if iid not in items:
                continue
            sense = (r.get("lev1_conn2") or "").strip()
            if not sense or sense.upper() == "NA":
                continue
            if "," in sense:  # multi-sense -> ambiguous, drop
                continue
            items[iid]["annotator_senses"].append(sense)
    print(f"Processed {n_rows} annotation rows")

    # Build training records
    out_rows = []
    n_excluded = 0
    n_too_few_ann = 0
    n_no_l1 = 0
    for iid, item in items.items():
        if iid in excluded:
            n_excluded += 1
            continue
        ann = item["annotator_senses"]
        if len(ann) < args.min_annotators:
            n_too_few_ann += 1
            continue
        counts = Counter(ann)
        l1_counts = {s: counts.get(s, 0) for s in SENSES}
        total = sum(l1_counts.values())
        if total == 0:
            n_no_l1 += 1
            continue
        dist = {s: l1_counts[s] / total for s in SENSES}
        out_rows.append({
            "item_id": iid,
            "arg1": item["arg1"],
            "arg2": item["arg2"],
            "genre": item["genre"],
            "n_annotators": len(ann),
            "label_distribution": dist,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {out_path}")
    print(f"  {len(out_rows)} pre-training items")
    print(f"  Excluded: {n_excluded} (in fine-tune set), {n_too_few_ann} (too few annotators), {n_no_l1} (no valid L1 votes)")

    # Genre breakdown
    genre_counts = Counter(r["genre"] for r in out_rows)
    print(f"  Genre breakdown (top 6):")
    for g, c in genre_counts.most_common(6):
        print(f"    {g or '<unknown>'}: {c}")


if __name__ == "__main__":
    main()
