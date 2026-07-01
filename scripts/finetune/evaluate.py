#!/usr/bin/env python3
"""Evaluate a fine-tuned model on the held-out test set.

Computes per-item predicted distributions, then KL/JSD against the crowd
distribution AND against the Wikipedia editor-gold distribution (the
genuinely-external signal). Writes a per-item CSV and prints aggregate
metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import model class from sibling train.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SENSES, SenseClassifier, pick_device  # noqa: E402


def kld(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    out = 0.0
    for s in SENSES:
        pv = p.get(s, 0.0)
        if pv > 0:
            qv = q.get(s, 0.0)
            out += pv * math.log((pv + eps) / (qv + eps))
    return out


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    m = {s: 0.5 * (p.get(s, 0.0) + q.get(s, 0.0)) for s in SENSES}
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)


def multilabel_f1(pred_set: set[str], gold_set: set[str]) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec = tp / len(gold_set)
    return 2 * prec * rec / (prec + rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--pred-threshold", type=float, default=0.30,
                        help="Probabilities above this become 'predicted' senses for the multilabel F1.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = json.load((model_dir / "config.json").open())
    device = pick_device()
    print(f"Device: {device}  model: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = SenseClassifier(config["backbone"]).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()
    max_length = config.get("max_length", 256)

    rows = []
    with open(args.test_jsonl) as f, torch.no_grad():
        for line in f:
            r = json.loads(line)
            enc = tokenizer(
                r["arg1"], r["arg2"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            pred = {s: float(probs[i]) for i, s in enumerate(SENSES)}
            top1_pred = max(pred, key=pred.get)
            pred_set = {s for s in SENSES if pred[s] >= args.pred_threshold}

            crowd = r["crowd_distribution"]
            row = {
                "item_id": r["item_id"],
                **{f"pred_{s}": round(pred[s], 4) for s in SENSES},
                **{f"crowd_{s}": round(crowd[s], 4) for s in SENSES},
                "top1_pred": top1_pred,
                "top1_crowd": max(crowd, key=crowd.get),
                "kl_vs_crowd": kld(crowd, pred),
                "jsd_vs_crowd": jsd(crowd, pred),
            }
            row["top1_match_crowd"] = int(row["top1_pred"] == row["top1_crowd"])

            gold = r.get("gold_distribution")
            if gold:
                row["kl_vs_gold"] = kld(gold, pred)
                row["jsd_vs_gold"] = jsd(gold, pred)
                gold_senses = set(r["gold_senses"])
                row["gold_senses"] = ";".join(sorted(gold_senses))
                row["top1_in_gold"] = int(top1_pred in gold_senses)
                row["multilabel_f1_gold"] = multilabel_f1(pred_set, gold_senses)
            rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"Wrote {out_csv} ({len(rows)} items)")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    gold_rows = [r for r in rows if "kl_vs_gold" in r]
    summary = {
        "n_test_items": len(rows),
        "n_with_gold": len(gold_rows),
        "kl_vs_crowd_mean": mean(r["kl_vs_crowd"] for r in rows),
        "jsd_vs_crowd_mean": mean(r["jsd_vs_crowd"] for r in rows),
        "top1_crowd_accuracy": mean(r["top1_match_crowd"] for r in rows),
    }
    if gold_rows:
        summary["kl_vs_gold_mean"] = mean(r["kl_vs_gold"] for r in gold_rows)
        summary["jsd_vs_gold_mean"] = mean(r["jsd_vs_gold"] for r in gold_rows)
        summary["top1_in_gold_accuracy"] = mean(r["top1_in_gold"] for r in gold_rows)
        summary["multilabel_f1_gold_mean"] = mean(r["multilabel_f1_gold"] for r in gold_rows)

    summary_path = out_csv.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
