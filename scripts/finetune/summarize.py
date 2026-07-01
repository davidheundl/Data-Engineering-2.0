#!/usr/bin/env python3
"""Aggregate per-run summary JSONs into a comparison table + bar plot.

Reads <results-dir>/eval_<backbone>_<variant>_seed<N>.summary.json files
and aggregates over seeds (mean ± std) per (backbone, variant). Writes:
  - summary_table.csv  (long-form per-seed numbers)
  - summary_table.md   (Markdown table with mean ± std)
  - summary_plot.png   (KL-vs-Gold + F1-vs-Gold bar chart with error bars)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANT_ORDER = ["raw_crowd", "raw_evade", "quadrant_curated"]
VARIANT_LABEL = {
    "raw_crowd": "Baseline A: Crowd",
    "raw_evade": "Baseline B: EVADE",
    "quadrant_curated": "Ours: Quadrant-curated",
}
COLORS = {
    "raw_crowd": "#9aa0a6",
    "raw_evade": "#f29900",
    "quadrant_curated": "#1a73e8",
}

METRICS_REPORT = [
    ("kl_vs_gold_mean", "KL vs Gold ↓", "lower"),
    ("jsd_vs_gold_mean", "JSD vs Gold ↓", "lower"),
    ("top1_in_gold_accuracy", "Top-1 in Gold ↑", "higher"),
    ("multilabel_f1_gold_mean", "Multilabel F1 ↑", "higher"),
    ("kl_vs_crowd_mean", "KL vs Crowd ↓", "lower"),
    ("top1_crowd_accuracy", "Top-1 Crowd ↑", "higher"),
]


def fmt(val: float, kind: str = "ratio") -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    if kind == "pct":
        return f"{val*100:.1f}%"
    return f"{val:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group seeds under (backbone, variant, pretrain). The trailing `_pt`
    # marks runs that were initialised from a pre-trained DiscoGeM checkpoint.
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    raw_rows = []
    pat = re.compile(
        r"eval_(.+?)_(raw_crowd|raw_evade|quadrant_curated)_seed(\d+)(_pt)?\.summary\.json"
    )
    for summary_path in sorted(results_dir.glob("eval_*.summary.json")):
        m = pat.match(summary_path.name)
        if not m:
            continue
        backbone, variant, seed = m.group(1), m.group(2), int(m.group(3))
        pretrain = "pretrained" if m.group(4) else "scratch"
        s = json.loads(summary_path.read_text())
        s["backbone"] = backbone
        s["variant"] = variant
        s["seed"] = seed
        s["pretrain"] = pretrain
        grouped[(backbone, variant, pretrain)].append(s)
        raw_rows.append(s)

    if not raw_rows:
        print(f"No summary files found in {results_dir}")
        return

    # Long-form CSV (one row per seed)
    csv_path = out_dir / "summary_table.csv"
    cols = ["backbone", "pretrain", "variant", "seed", "n_test_items", "n_with_gold",
            "kl_vs_crowd_mean", "jsd_vs_crowd_mean", "top1_crowd_accuracy",
            "kl_vs_gold_mean", "jsd_vs_gold_mean", "top1_in_gold_accuracy",
            "multilabel_f1_gold_mean"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(raw_rows, key=lambda r: (r["backbone"], r.get("pretrain", "scratch"), VARIANT_ORDER.index(r["variant"]), r["seed"])):
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Aggregate per (backbone, variant, pretrain)
    agg = {}
    for key, runs in grouped.items():
        d = {}
        d["n_seeds"] = len(runs)
        d["seeds"] = sorted(r["seed"] for r in runs)
        for metric, _label, _kind in METRICS_REPORT:
            vals = [r.get(metric) for r in runs if r.get(metric) is not None]
            if not vals:
                continue
            d[f"{metric}_mean"] = st.mean(vals)
            d[f"{metric}_std"] = st.stdev(vals) if len(vals) > 1 else 0.0
        agg[key] = d

    # Markdown
    md_lines = []
    headers = ["Backbone", "Init", "Train Labels", "Seeds"]
    headers += [m[1] for m in METRICS_REPORT]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    backbones = sorted({b for (b, _, _) in agg.keys()})
    pretrain_order = ["scratch", "pretrained"]
    pretrain_label = {"scratch": "from scratch", "pretrained": "pre-trained"}
    for bb in backbones:
        for pretrain in pretrain_order:
            for variant in VARIANT_ORDER:
                d = agg.get((bb, variant, pretrain))
                if d is None:
                    continue
                row = [bb, pretrain_label[pretrain], VARIANT_LABEL[variant], f"{d['n_seeds']}"]
                for metric, _label, _kind in METRICS_REPORT:
                    m = d.get(f"{metric}_mean")
                    sd = d.get(f"{metric}_std", 0.0)
                    if m is None:
                        row.append("—")
                    else:
                        is_pct = "accuracy" in metric
                        if is_pct:
                            row.append(f"{m*100:.1f}±{sd*100:.1f}%")
                        else:
                            row.append(f"{m:.3f}±{sd:.3f}")
                md_lines.append("| " + " | ".join(row) + " |")

    md_path = out_dir / "summary_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    # Plot: KL-vs-Gold + Multilabel-F1 bar chart with error bars. If both
    # scratch and pretrained runs exist, we plot two subplots per metric row.
    pretrain_groups = sorted({pt for (_, _, pt) in agg.keys()}, key=lambda x: pretrain_order.index(x))
    n_rows = len(pretrain_groups)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows), squeeze=False)
    for row_idx, pretrain in enumerate(pretrain_groups):
        for col_idx, (metric, title) in enumerate([
            ("kl_vs_gold_mean", "KL Divergence vs Wikipedia Gold (lower = better)"),
            ("multilabel_f1_gold_mean", "Multilabel F1 vs Wikipedia Gold (higher = better)"),
        ]):
            ax = axes[row_idx][col_idx]
            x = np.arange(len(backbones))
            width = 0.25
            for i, variant in enumerate(VARIANT_ORDER):
                means, stds = [], []
                for bb in backbones:
                    d = agg.get((bb, variant, pretrain), {})
                    m_val = d.get(f"{metric}_mean")
                    sd_val = d.get(f"{metric}_std", 0.0)
                    means.append(m_val if m_val is not None else float("nan"))
                    stds.append(sd_val)
                offsets = x + (i - 1) * width
                bars = ax.bar(offsets, means, width, yerr=stds, capsize=4,
                              label=VARIANT_LABEL[variant],
                              color=COLORS[variant], edgecolor="black", linewidth=0.5,
                              error_kw={"linewidth": 1, "ecolor": "#333"})
                for b, mn, sd in zip(bars, means, stds):
                    if mn == mn:
                        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.005,
                                f"{mn:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(backbones)
            ax.set_title(f"{pretrain_label[pretrain]}: {title}", fontsize=11)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plot_path = out_dir / "summary_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Wrote {plot_path}")

    # Plain-text table to stdout
    print()
    print(md_path.read_text())


if __name__ == "__main__":
    main()
