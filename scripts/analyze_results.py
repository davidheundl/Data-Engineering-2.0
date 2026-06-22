#!/usr/bin/env python3
"""Comprehensive analysis of Level-1 EVADE pipeline results.

Standalone script that produces publication-quality plots and metric tables
comparing LLM-derived sense distributions against human crowd annotations.

Usage:
    python scripts/analyze_results.py --run-dir results/20260607T081000Z_level1_experiment_7108c04
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate import _kld, read_distributions
from src.generate import read_generations
from src.prep import read_items
from src.schemas import (
    DistributionRecord,
    GenerationRecord,
    PrepItem,
    ValidationRecord,
)
from src.validate import read_validations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENSES = ["temporal", "contingency", "comparison", "expansion"]
GENRE_COLORS = {"Europarl": "#4C72B0", "Lit": "#DD8452", "Wiki": "#55A868"}
SENSE_COLORS = {"temporal": "#4C72B0", "contingency": "#DD8452",
                "comparison": "#55A868", "expansion": "#C44E52"}
MODEL_SHORT = {
    "openai:gpt-4o-mini": "GPT-4o-mini",
    "anthropic:claude-haiku-4-5-20251001": "Claude Haiku",
    "mistral:mistral-small-latest": "Mistral Small",
    "deepseek:deepseek-chat": "DeepSeek Chat",
}
MODEL_COLORS = {
    "openai:gpt-4o-mini": "#4C72B0",
    "anthropic:claude-haiku-4-5-20251001": "#DD8452",
    "mistral:mistral-small-latest": "#55A868",
    "deepseek:deepseek-chat": "#C44E52",
}

# Matplotlib defaults for publication quality
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_crowd(dist: dict[str, float]) -> dict[str, float]:
    """Remove norel and renormalize crowd distribution to 4 senses."""
    filtered = {s: dist.get(s, 0.0) for s in SENSES}
    total = sum(filtered.values())
    if total <= 0:
        return {s: 0.25 for s in SENSES}
    return {s: v / total for s, v in filtered.items()}


def _get_llm_dist(rec: DistributionRecord, tau: str = "0.30") -> dict[str, float]:
    """Extract LLM distribution at given tau, filling missing senses with 0."""
    raw = rec.llm_label_distribution_per_tau.get(tau, {})
    dist = {s: raw.get(s, 0.0) for s in SENSES}
    total = sum(dist.values())
    if total <= 0:
        return {s: 0.25 for s in SENSES}
    return {s: v / total for s, v in dist.items()}


def load_comparative_results(path: Path) -> list[dict]:
    """Load comparative_results.jsonl (raw dicts, no Pydantic model)."""
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon Divergence using natural log."""
    m = {s: 0.5 * (p.get(s, 0.0) + q.get(s, 0.0)) for s in set(p) | set(q)}
    return 0.5 * _kld(p, m) + 0.5 * _kld(q, m)


def mae(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys) / len(keys)


def rmse(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return math.sqrt(sum((p.get(k, 0.0) - q.get(k, 0.0))**2 for k in keys) / len(keys))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx)**2 for x in xs) / n)
    sy = math.sqrt(sum((y - my)**2 for y in ys) / n)
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation."""
    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks
    return pearson_r(_rank(xs), _rank(ys))


def compute_per_item_metrics(
    items: list[PrepItem],
    dists_a: list[DistributionRecord],
    dists_b: list[DistributionRecord],
    tau: str = "0.30",
) -> list[dict]:
    """Compute all metrics per item for both variants."""
    items_by_id = {it.item_id: it for it in items}
    dists_a_by_id = {d.item_id: d for d in dists_a}
    dists_b_by_id = {d.item_id: d for d in dists_b}

    rows = []
    for item in items:
        crowd = _normalize_crowd(item.crowd_sense_distribution)
        crowd_top = max(crowd, key=crowd.get)
        n_crowd_senses = sum(1 for v in crowd.values() if v > 0.01)

        row = {
            "item_id": item.item_id,
            "genre": item.genre,
            "stratification_bin": item.stratification_bin,
            "crowd_agreement": item.crowd_agreement_score,
            "crowd_majority": crowd_top,
            "n_crowd_senses": n_crowd_senses,
            "n_valid_annotations": item.n_valid_annotations,
        }

        for label, dists_map in [("A", dists_a_by_id), ("B", dists_b_by_id)]:
            rec = dists_map.get(item.item_id)
            if rec is None:
                continue
            llm = _get_llm_dist(rec, tau)
            llm_top = max(llm, key=llm.get)
            row[f"jsd_{label}"] = jsd(crowd, llm)
            row[f"kld_{label}"] = _kld(crowd, llm)
            row[f"mae_{label}"] = mae(crowd, llm)
            row[f"rmse_{label}"] = rmse(crowd, llm)
            row[f"top1_match_{label}"] = crowd_top == llm_top
            row[f"llm_top1_{label}"] = llm_top
            row[f"llm_top1_prob_{label}"] = llm[llm_top]
            for s in SENSES:
                row[f"crowd_{s}"] = crowd[s]
                row[f"llm_{label}_{s}"] = llm[s]

        rows.append(row)
    return rows


def compute_fleiss_kappa(comp_results: list[dict], items: list[PrepItem]) -> float:
    """Fleiss' kappa for top-1 sense agreement across models."""
    item_ids = sorted(set(r["item_id"] for r in comp_results))
    models = sorted(set(r["model"] for r in comp_results))
    n_raters = len(models)
    n_categories = len(SENSES)

    # Build rating matrix: items x senses (count of raters choosing each sense)
    ratings = []
    for iid in item_ids:
        row = {s: 0 for s in SENSES}
        for r in comp_results:
            if r["item_id"] == iid:
                dist = r["distribution"]
                top = max(dist, key=dist.get)
                row[top] += 1
        ratings.append([row[s] for s in SENSES])

    N = len(ratings)
    if N == 0 or n_raters < 2:
        return 0.0

    # Fleiss' kappa
    p_j = [sum(row[j] for row in ratings) / (N * n_raters) for j in range(n_categories)]
    P_i = [(sum(n_ij**2 for n_ij in row) - n_raters) / (n_raters * (n_raters - 1))
           for row in ratings]
    P_bar = sum(P_i) / N
    P_e = sum(p**2 for p in p_j)

    if abs(1 - P_e) < 1e-10:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ---------------------------------------------------------------------------
# A: LLM vs Crowd plots
# ---------------------------------------------------------------------------
def plot_jsd_per_item(rows: list[dict], out_dir: Path):
    """A1: JSD per item bar chart, sorted, colored by genre."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, label, title in [(axes[0], "A", "Variante A (Aggregate)"),
                             (axes[1], "B", "Variante B (Compare)")]:
        sorted_rows = sorted(rows, key=lambda r: r.get(f"jsd_{label}", 0))
        x = range(len(sorted_rows))
        colors = [GENRE_COLORS[r["genre"]] for r in sorted_rows]
        vals = [r.get(f"jsd_{label}", 0) for r in sorted_rows]
        ax.bar(x, vals, color=colors, width=0.8, edgecolor="none")
        mean_jsd = mean(vals) if vals else 0
        ax.axhline(mean_jsd, color="black", ls="--", lw=1, alpha=0.7,
                    label=f"mean = {mean_jsd:.3f}")
        ax.set_title(title)
        ax.set_xlabel("Items (sorted by JSD)")
        ax.set_ylabel("Jensen-Shannon Divergence")
        ax.set_xticks([])
        ax.legend(loc="upper left")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
               markersize=10, label=g) for g, c in GENRE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True)
    fig.suptitle("JSD per Item: LLM vs Crowd Distribution", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(out_dir / "jsd_per_item_barplot.png")
    plt.close()


def plot_correlation_scatter(rows: list[dict], out_dir: Path):
    """A2: 2x2 scatter grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # (a) crowd prob vs LLM prob, colored by sense (Variante B)
    ax = axes[0, 0]
    for s in SENSES:
        cx = [r[f"crowd_{s}"] for r in rows if f"llm_B_{s}" in r]
        ly = [r[f"llm_B_{s}"] for r in rows if f"llm_B_{s}" in r]
        ax.scatter(cx, ly, c=SENSE_COLORS[s], label=s, alpha=0.6, s=40, edgecolors="none")
    all_cx = [r[f"crowd_{s}"] for r in rows for s in SENSES if f"llm_B_{s}" in r]
    all_ly = [r[f"llm_B_{s}"] for r in rows for s in SENSES if f"llm_B_{s}" in r]
    r_val = pearson_r(all_cx, all_ly)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Crowd probability")
    ax.set_ylabel("LLM probability")
    ax.set_title(f"(a) By Sense (Var. B, r={r_val:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (b) crowd prob vs LLM prob, colored by genre (Variante B)
    ax = axes[0, 1]
    for genre, color in GENRE_COLORS.items():
        genre_rows = [r for r in rows if r["genre"] == genre]
        cx = [r[f"crowd_{s}"] for r in genre_rows for s in SENSES if f"llm_B_{s}" in r]
        ly = [r[f"llm_B_{s}"] for r in genre_rows for s in SENSES if f"llm_B_{s}" in r]
        ax.scatter(cx, ly, c=color, label=genre, alpha=0.6, s=40, edgecolors="none")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Crowd probability")
    ax.set_ylabel("LLM probability")
    ax.set_title(f"(b) By Genre (Var. B, r={r_val:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (c) crowd_agreement vs JSD
    ax = axes[1, 0]
    for genre, color in GENRE_COLORS.items():
        gr = [r for r in rows if r["genre"] == genre and "jsd_B" in r]
        ax.scatter([r["crowd_agreement"] for r in gr],
                   [r["jsd_B"] for r in gr],
                   c=color, label=genre, alpha=0.7, s=50, edgecolors="none")
    agr = [r["crowd_agreement"] for r in rows if "jsd_B" in r]
    jsd_vals = [r["jsd_B"] for r in rows if "jsd_B" in r]
    rho = spearman_rho(agr, jsd_vals) if len(agr) > 2 else 0
    ax.set_xlabel("Crowd Agreement Score")
    ax.set_ylabel("JSD (Variante B)")
    ax.set_title(f"(c) Agreement vs JSD (ρ={rho:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (d) n_crowd_senses vs JSD
    ax = axes[1, 1]
    for genre, color in GENRE_COLORS.items():
        gr = [r for r in rows if r["genre"] == genre and "jsd_B" in r]
        ax.scatter([r["n_crowd_senses"] for r in gr],
                   [r["jsd_B"] for r in gr],
                   c=color, label=genre, alpha=0.7, s=50, edgecolors="none")
    ax.set_xlabel("Number of crowd senses (>1%)")
    ax.set_ylabel("JSD (Variante B)")
    n_senses = [r["n_crowd_senses"] for r in rows if "jsd_B" in r]
    rho2 = spearman_rho(n_senses, jsd_vals) if len(n_senses) > 2 else 0
    ax.set_title(f"(d) #Senses vs JSD (ρ={rho2:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle("Correlation Analysis: Crowd vs LLM", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "correlation_scatter.png")
    plt.close()


def _covariance(xs: list[float], ys: list[float]) -> float:
    """Population covariance between xs and ys."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n


def plot_crowd_vs_llm_covariance(
    items: list[PrepItem],
    dists_b: list[DistributionRecord],
    out_dir: Path,
):
    """Standalone scatter: crowd prob vs LLM prob (Var. B) by genre, across taus."""
    taus = ["0.30", "0.40", "0.50", "0.60", "0.70"]
    dists_by_id = {d.item_id: d for d in dists_b}

    fig, axes = plt.subplots(1, len(taus), figsize=(25, 5.5))

    for idx, tau in enumerate(taus):
        ax = axes[idx]
        all_cx, all_ly = [], []
        genre_data: dict[str, tuple[list[float], list[float]]] = {
            g: ([], []) for g in GENRE_COLORS
        }

        for item in items:
            rec = dists_by_id.get(item.item_id)
            if rec is None:
                continue
            crowd = _normalize_crowd(item.crowd_sense_distribution)
            llm = _get_llm_dist(rec, tau)
            for s in SENSES:
                cx, ly = crowd[s], llm[s]
                all_cx.append(cx)
                all_ly.append(ly)
                genre_data[item.genre][0].append(cx)
                genre_data[item.genre][1].append(ly)

        for genre, color in GENRE_COLORS.items():
            gx, gy = genre_data[genre]
            if gx:
                ax.scatter(gx, gy, c=color, label=genre, alpha=0.4, s=30,
                           edgecolors="none")

        ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.4, lw=0.8)

        # Overall regression line
        all_cx_arr, all_ly_arr = np.array(all_cx), np.array(all_ly)
        coeffs_all = np.polyfit(all_cx_arr, all_ly_arr, 1)
        x_fit = np.linspace(0, 1, 100)
        y_fit = np.polyval(coeffs_all, x_fit)
        ax.plot(x_fit, y_fit, color="black", lw=2, alpha=0.8)

        ax.set_xlabel("Crowd probability")
        if idx == 0:
            ax.set_ylabel("LLM probability (Var. B)")
        ax.set_title(f"τ = {tau}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Crowd vs LLM Sense Distribution by Genre (Variante B)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_dir / "crowd_vs_llm_by_genre_covariance.png", dpi=150)
    plt.close()
    print(f"  Wrote {out_dir / 'crowd_vs_llm_by_genre_covariance.png'}")


def plot_calibration(rows: list[dict], out_dir: Path):
    """A3: Reliability diagram."""
    fig, ax = plt.subplots(figsize=(7, 6))
    n_bins = 10

    for label, color, marker in [("A", "#4C72B0", "o"), ("B", "#DD8452", "s")]:
        preds, actuals = [], []
        for r in rows:
            for s in SENSES:
                if f"llm_{label}_{s}" in r:
                    preds.append(r[f"llm_{label}_{s}"])
                    actuals.append(r[f"crowd_{s}"])

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means_pred, bin_means_actual, bin_counts = [], [], []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = [(lo <= p < hi) or (i == n_bins - 1 and p == hi)
                    for p in preds]
            bin_p = [p for p, m in zip(preds, mask) if m]
            bin_a = [a for a, m in zip(actuals, mask) if m]
            if bin_p:
                bin_means_pred.append(mean(bin_p))
                bin_means_actual.append(mean(bin_a))
                bin_counts.append(len(bin_p))
            else:
                bin_means_pred.append((lo + hi) / 2)
                bin_means_actual.append(0)
                bin_counts.append(0)

        ax.plot(bin_means_pred, bin_means_actual, f"-{marker}",
                color=color, label=f"Variante {label}", markersize=6)

    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean LLM predicted probability")
    ax.set_ylabel("Mean crowd probability")
    ax.set_title("Calibration: Reliability Diagram")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(out_dir / "calibration_diagram.png")
    plt.close()


def plot_per_sense_bias(rows: list[dict], out_dir: Path):
    """A4: Per-sense mean crowd vs LLM probability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, label, title in [(axes[0], "A", "Variante A (Aggregate)"),
                             (axes[1], "B", "Variante B (Compare)")]:
        x = np.arange(len(SENSES))
        width = 0.35
        crowd_means = [mean(r[f"crowd_{s}"] for r in rows) for s in SENSES]
        llm_means = [mean(r[f"llm_{label}_{s}"] for r in rows if f"llm_{label}_{s}" in r)
                     for s in SENSES]
        crowd_stds = [stdev(r[f"crowd_{s}"] for r in rows) if len(rows) > 1 else 0
                      for s in SENSES]
        llm_stds = [stdev(r[f"llm_{label}_{s}"] for r in rows if f"llm_{label}_{s}" in r)
                    if sum(1 for r in rows if f"llm_{label}_{s}" in r) > 1 else 0
                    for s in SENSES]

        ax.bar(x - width / 2, crowd_means, width, yerr=crowd_stds, label="Crowd",
               color="#7FB3D8", capsize=3)
        ax.bar(x + width / 2, llm_means, width, yerr=llm_stds, label="LLM",
               color="#F4A582", capsize=3)

        # Annotate bias
        for i, s in enumerate(SENSES):
            bias = llm_means[i] - crowd_means[i]
            ax.text(i, max(crowd_means[i], llm_means[i]) + max(crowd_stds[i], llm_stds[i]) + 0.02,
                    f"{bias:+.3f}", ha="center", fontsize=8, color="red" if bias > 0 else "blue")

        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in SENSES])
        ax.set_ylabel("Mean probability")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle("Per-Sense Bias: Crowd vs LLM Mean Probability", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "per_sense_bias.png")
    plt.close()


def plot_confusion_matrix(rows: list[dict], out_dir: Path):
    """A5: Top-1 confusion matrix for both variants."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label, title in [(axes[0], "A", "Variante A (Aggregate)"),
                             (axes[1], "B", "Variante B (Compare)")]:
        mat = np.zeros((4, 4), dtype=int)
        for r in rows:
            if f"llm_top1_{label}" not in r:
                continue
            true_idx = SENSES.index(r["crowd_majority"])
            pred_idx = SENSES.index(r[f"llm_top1_{label}"])
            mat[true_idx, pred_idx] += 1

        # Row-normalize for display
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.where(row_sums > 0, mat / row_sums, 0)

        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([s[:4].capitalize() for s in SENSES], rotation=45, ha="right")
        ax.set_yticklabels([s[:4].capitalize() for s in SENSES])
        ax.set_xlabel("LLM Top-1")
        ax.set_ylabel("Crowd Top-1")

        # Annotate cells with count and percentage
        for i in range(4):
            for j in range(4):
                txt = f"{mat[i, j]}\n({mat_norm[i, j]:.0%})"
                color = "white" if mat_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

        acc = sum(r.get(f"top1_match_{label}", False) for r in rows) / len(rows)
        ax.set_title(f"{title}\nTop-1 Acc: {acc:.1%}")

    fig.suptitle("Confusion Matrix: Crowd vs LLM Top-1 Sense", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_dir / "confusion_matrix_top1.png")
    plt.close()


# ---------------------------------------------------------------------------
# B: Between categories
# ---------------------------------------------------------------------------
def plot_genre_boxplot(rows: list[dict], out_dir: Path):
    """B1: JSD by genre boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, label, title in [(axes[0], "A", "Variante A"), (axes[1], "B", "Variante B")]:
        data = []
        labels_g = []
        colors = []
        for genre in ["Europarl", "Lit", "Wiki"]:
            vals = [r[f"jsd_{label}"] for r in rows if r["genre"] == genre and f"jsd_{label}" in r]
            data.append(vals)
            labels_g.append(f"{genre}\n(n={len(vals)})")
            colors.append(GENRE_COLORS[genre])

        bp = ax.boxplot(data, labels=labels_g, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("JSD")
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle("JSD by Genre", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "jsd_by_genre_boxplot.png")
    plt.close()


def plot_sense_genre_heatmap(rows: list[dict], out_dir: Path):
    """B2: MAE per (Sense x Genre) heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    genres = ["Europarl", "Lit", "Wiki"]

    for ax, label, title in [(axes[0], "A", "Variante A"), (axes[1], "B", "Variante B")]:
        mat = np.zeros((len(SENSES), len(genres)))
        for i, s in enumerate(SENSES):
            for j, g in enumerate(genres):
                gr = [r for r in rows if r["genre"] == g and f"llm_{label}_{s}" in r]
                if gr:
                    mat[i, j] = mean(abs(r[f"crowd_{s}"] - r[f"llm_{label}_{s}"]) for r in gr)

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, aspect="auto")
        ax.set_xticks(range(len(genres)))
        ax.set_yticks(range(len(SENSES)))
        ax.set_xticklabels(genres)
        ax.set_yticklabels([s.capitalize() for s in SENSES])
        for i in range(len(SENSES)):
            for j in range(len(genres)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="MAE")

    fig.suptitle("MAE per Sense x Genre", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "per_sense_genre_heatmap.png")
    plt.close()


def plot_confusion_by_genre(rows: list[dict], out_dir: Path):
    """B3: Confusion matrices per genre (Variante B only)."""
    genres = ["Europarl", "Lit", "Wiki"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, genre in zip(axes, genres):
        gr = [r for r in rows if r["genre"] == genre and "llm_top1_B" in r]
        mat = np.zeros((4, 4), dtype=int)
        for r in gr:
            true_idx = SENSES.index(r["crowd_majority"])
            pred_idx = SENSES.index(r["llm_top1_B"])
            mat[true_idx, pred_idx] += 1

        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.where(row_sums > 0, mat / row_sums, 0)

        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([s[:4].capitalize() for s in SENSES], rotation=45, ha="right")
        ax.set_yticklabels([s[:4].capitalize() for s in SENSES])
        for i in range(4):
            for j in range(4):
                txt = f"{mat[i, j]}"
                color = "white" if mat_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)
        acc = sum(1 for r in gr if r.get("top1_match_B")) / max(len(gr), 1)
        ax.set_title(f"{genre} (n={len(gr)}, acc={acc:.0%})")
        ax.set_xlabel("LLM Top-1")
        ax.set_ylabel("Crowd Top-1")

    fig.suptitle("Confusion Matrix by Genre (Variante B)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "confusion_by_genre.png")
    plt.close()


def plot_agreement_vs_jsd(rows: list[dict], out_dir: Path):
    """B4: Crowd agreement vs JSD scatter with regression."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for genre, color in GENRE_COLORS.items():
        gr = [r for r in rows if r["genre"] == genre and "jsd_B" in r]
        ax.scatter([r["crowd_agreement"] for r in gr],
                   [r["jsd_B"] for r in gr],
                   c=color, label=genre, s=60, alpha=0.7, edgecolors="none")

    xs = [r["crowd_agreement"] for r in rows if "jsd_B" in r]
    ys = [r["jsd_B"] for r in rows if "jsd_B" in r]

    # Regression line
    if len(xs) > 2:
        coeffs = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "--", color="gray", alpha=0.7)
        rho = spearman_rho(xs, ys)
        ax.set_title(f"Crowd Agreement vs JSD (Spearman ρ = {rho:.3f})")

    ax.set_xlabel("Crowd Agreement Score")
    ax.set_ylabel("JSD (Variante B)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.savefig(out_dir / "agreement_vs_jsd_scatter.png")
    plt.close()


# ---------------------------------------------------------------------------
# C: Between LLMs
# ---------------------------------------------------------------------------
def plot_model_distributions(comp_results: list[dict], out_dir: Path):
    """C1: Mean distribution per model."""
    models = sorted(set(r["model"] for r in comp_results))
    model_dists = {m: {s: [] for s in SENSES} for m in models}
    for r in comp_results:
        for s in SENSES:
            model_dists[r["model"]][s].append(r["distribution"].get(s, 0.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SENSES))
    width = 0.18
    offsets = np.arange(len(models)) * width - width * (len(models) - 1) / 2

    for i, m in enumerate(models):
        means = [mean(model_dists[m][s]) for s in SENSES]
        stds = [stdev(model_dists[m][s]) if len(model_dists[m][s]) > 1 else 0 for s in SENSES]
        ax.bar(x + offsets[i], means, width, yerr=stds, label=MODEL_SHORT.get(m, m),
               color=MODEL_COLORS.get(m, "gray"), capsize=2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SENSES])
    ax.set_ylabel("Mean probability")
    ax.set_title("Mean Sense Distribution per Model (Comparative)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.savefig(out_dir / "per_model_mean_distribution.png")
    plt.close()


def plot_model_jsd_boxplot(items: list[PrepItem], comp_results: list[dict], out_dir: Path):
    """C2: JSD per model boxplot."""
    items_by_id = {it.item_id: it for it in items}
    models = sorted(set(r["model"] for r in comp_results))

    model_jsds = {m: [] for m in models}
    for r in comp_results:
        item = items_by_id.get(r["item_id"])
        if not item:
            continue
        crowd = _normalize_crowd(item.crowd_sense_distribution)
        llm = {s: r["distribution"].get(s, 0.0) for s in SENSES}
        model_jsds[r["model"]].append(jsd(crowd, llm))

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [model_jsds[m] for m in models]
    labels = [MODEL_SHORT.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "gray") for m in models]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add mean annotations
    for i, m in enumerate(models):
        if model_jsds[m]:
            ax.text(i + 1, mean(model_jsds[m]) + 0.01,
                    f"μ={mean(model_jsds[m]):.3f}", ha="center", fontsize=8)

    ax.set_ylabel("JSD vs Crowd")
    ax.set_title("Per-Model JSD: Individual Model vs Crowd Distribution")
    ax.grid(alpha=0.2, axis="y")
    plt.savefig(out_dir / "per_model_jsd_boxplot.png")
    plt.close()


def plot_inter_model_agreement(comp_results: list[dict], out_dir: Path):
    """C3: Pairwise model agreement heatmap."""
    models = sorted(set(r["model"] for r in comp_results))
    item_ids = sorted(set(r["item_id"] for r in comp_results))
    n = len(models)

    # Build lookup: (item_id, model) -> distribution
    lookup = {}
    for r in comp_results:
        lookup[(r["item_id"], r["model"])] = r["distribution"]

    # Pairwise top-1 agreement and JSD
    agree_mat = np.zeros((n, n))
    jsd_mat = np.zeros((n, n))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            agrees = 0
            jsds = []
            count = 0
            for iid in item_ids:
                d1 = lookup.get((iid, m1))
                d2 = lookup.get((iid, m2))
                if d1 and d2:
                    t1 = max(d1, key=d1.get)
                    t2 = max(d2, key=d2.get)
                    agrees += (t1 == t2)
                    p = {s: d1.get(s, 0.0) for s in SENSES}
                    q = {s: d2.get(s, 0.0) for s in SENSES}
                    jsds.append(jsd(p, q))
                    count += 1
            agree_mat[i, j] = agrees / max(count, 1)
            jsd_mat[i, j] = mean(jsds) if jsds else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    short_names = [MODEL_SHORT.get(m, m) for m in models]

    # Top-1 agreement
    ax = axes[0]
    im = ax.imshow(agree_mat, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agree_mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Top-1 Agreement Rate")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Mean JSD between models
    ax = axes[1]
    im = ax.imshow(jsd_mat, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jsd_mat[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mean JSD Between Models")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Inter-Model Agreement", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "inter_model_agreement.png")
    plt.close()


def plot_validator_generator_quality(
    gens: list[GenerationRecord],
    vals: list[ValidationRecord],
    out_dir: Path,
):
    """C4: Validator stringency + generator quality."""
    # Validator stringency: mean score given per validator model
    val_scores = defaultdict(list)
    for v in vals:
        val_scores[v.validator_model].append(v.validity_score)

    # Generator quality: mean score received per generator model
    gen_id_to_model = {g.generation_id: g.generator_model for g in gens}
    gen_scores = defaultdict(list)
    for v in vals:
        gm = gen_id_to_model.get(v.generation_id)
        if gm:
            gen_scores[gm].append(v.validity_score)

    models = sorted(set(val_scores.keys()) | set(gen_scores.keys()))
    short_names = [MODEL_SHORT.get(m, m) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Validator stringency
    ax = axes[0]
    means_v = [mean(val_scores[m]) if val_scores[m] else 0 for m in models]
    colors_v = [MODEL_COLORS.get(m, "gray") for m in models]
    bars = ax.bar(range(len(models)), means_v, color=colors_v, alpha=0.8)
    for i, v in enumerate(means_v):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=20, ha="right")
    ax.set_ylabel("Mean validity score given")
    ax.set_title("Validator Stringency")
    ax.grid(alpha=0.2, axis="y")

    # Generator quality
    ax = axes[1]
    means_g = [mean(gen_scores[m]) if gen_scores[m] else 0 for m in models]
    bars = ax.bar(range(len(models)), means_g, color=colors_v, alpha=0.8)
    for i, v in enumerate(means_g):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=20, ha="right")
    ax.set_ylabel("Mean validity score received")
    ax.set_title("Generator Quality")
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle("Validator Stringency & Generator Quality", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "validator_generator_quality.png")
    plt.close()


def plot_abstention_heatmap(gens: list[GenerationRecord], out_dir: Path):
    """C5: Abstention count per model x sense."""
    models = sorted(set(g.generator_model for g in gens))
    mat = np.zeros((len(models), len(SENSES)), dtype=int)

    for g in gens:
        if g.abstained:
            i = models.index(g.generator_model)
            if g.candidate_sense in SENSES:
                j = SENSES.index(g.candidate_sense)
                mat[i, j] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(SENSES)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([s.capitalize() for s in SENSES])
    ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in models])

    for i in range(len(models)):
        for j in range(len(SENSES)):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=11, color="white" if mat[i, j] > 3 else "black")

    ax.set_title("Abstention Count per Model x Sense")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    plt.tight_layout()
    plt.savefig(out_dir / "abstention_heatmap.png")
    plt.close()


# ---------------------------------------------------------------------------
# D: Variante A vs B
# ---------------------------------------------------------------------------
def plot_variante_scatter(rows: list[dict], out_dir: Path):
    """D1: JSD_A vs JSD_B scatter."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for genre, color in GENRE_COLORS.items():
        gr = [r for r in rows if r["genre"] == genre and "jsd_A" in r and "jsd_B" in r]
        ax.scatter([r["jsd_A"] for r in gr], [r["jsd_B"] for r in gr],
                   c=color, label=genre, s=60, alpha=0.7, edgecolors="none")

    lim = max(max(r.get("jsd_A", 0) for r in rows), max(r.get("jsd_B", 0) for r in rows)) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5)
    ax.set_xlabel("JSD — Variante A (Aggregate)")
    ax.set_ylabel("JSD — Variante B (Compare)")
    ax.set_title("Variante A vs B: Per-Item JSD")
    ax.legend()
    ax.grid(alpha=0.2)

    # Count which is better
    n_a_better = sum(1 for r in rows if r.get("jsd_A", 1) < r.get("jsd_B", 1))
    n_b_better = sum(1 for r in rows if r.get("jsd_B", 1) < r.get("jsd_A", 1))
    ax.text(0.05, 0.95, f"A better: {n_a_better}  |  B better: {n_b_better}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.savefig(out_dir / "variante_jsd_scatter.png")
    plt.close()


def plot_variante_comparison(rows: list[dict], out_dir: Path):
    """D2: Aggregate metrics comparison bar chart."""
    metrics = {}
    for label in ["A", "B"]:
        jsd_vals = [r[f"jsd_{label}"] for r in rows if f"jsd_{label}" in r]
        mae_vals = [r[f"mae_{label}"] for r in rows if f"mae_{label}" in r]
        rmse_vals = [r[f"rmse_{label}"] for r in rows if f"rmse_{label}" in r]
        top1 = [r[f"top1_match_{label}"] for r in rows if f"top1_match_{label}" in r]

        all_cx = [r[f"crowd_{s}"] for r in rows for s in SENSES if f"llm_{label}_{s}" in r]
        all_ly = [r[f"llm_{label}_{s}"] for r in rows for s in SENSES if f"llm_{label}_{s}" in r]

        metrics[label] = {
            "Mean JSD": mean(jsd_vals) if jsd_vals else 0,
            "Mean MAE": mean(mae_vals) if mae_vals else 0,
            "Mean RMSE": mean(rmse_vals) if rmse_vals else 0,
            "Top-1 Acc": sum(top1) / len(top1) if top1 else 0,
            "Pearson r": pearson_r(all_cx, all_ly),
            "Spearman ρ": spearman_rho(all_cx, all_ly),
        }

    metric_names = list(metrics["A"].keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names))
    width = 0.3

    vals_a = [metrics["A"][m] for m in metric_names]
    vals_b = [metrics["B"][m] for m in metric_names]

    ax.bar(x - width / 2, vals_a, width, label="Variante A (Aggregate)", color="#4C72B0", alpha=0.8)
    ax.bar(x + width / 2, vals_b, width, label="Variante B (Compare)", color="#DD8452", alpha=0.8)

    for i in range(len(metric_names)):
        ax.text(i - width / 2, vals_a[i] + 0.01, f"{vals_a[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, vals_b[i] + 0.01, f"{vals_b[i]:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title("Variante A vs B: Aggregate Metrics")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "variante_comparison_bars.png")
    plt.close()


# ---------------------------------------------------------------------------
# E: Risk quadrants (crowd agreement x LLM top-1 confidence)
# ---------------------------------------------------------------------------
QUADRANT_LABELS = {
    ("low", "low"): "high_risk",
    ("low", "high"): "llm_overconfident",
    ("high", "low"): "llm_underconfident",
    ("high", "high"): "safe",
}
QUADRANT_COLORS = {
    "high_risk": "#C44E52",
    "llm_overconfident": "#DD8452",
    "llm_underconfident": "#4C72B0",
    "safe": "#55A868",
}


def build_sense_rows(rows: list[dict]) -> list[dict]:
    """Expand per-item rows into one row per (item, sense), max 45 x 4 = 180.

    Senses with zero crowd probability are excluded: if no human ever chose
    the sense, there is no crowd judgment to compare against, so calling the
    LLM's score an error (or non-error) there is not meaningful.

    Each row also carries the raw annotator vote count for that sense
    (recovered from crowd_prob * n_valid_annotations). Crowd-axis binning
    uses this integer count so that items with different annotator pool
    sizes (9 vs 10 vs 11) are treated consistently - one annotator is one
    annotator regardless of the normalization denominator.
    """
    sense_rows = []
    for r in rows:
        n_ann = r.get("n_valid_annotations") or 0
        for s in SENSES:
            if f"crowd_{s}" not in r or r[f"crowd_{s}"] <= 0:
                continue
            crowd_prob = r[f"crowd_{s}"]
            crowd_count = int(round(crowd_prob * n_ann)) if n_ann else 0
            sr = {
                "item_id": r["item_id"],
                "genre": r["genre"],
                "sense": s,
                "crowd_agreement": r["crowd_agreement"],
                "crowd_majority": r["crowd_majority"],
                "crowd_prob": crowd_prob,
                "crowd_count": crowd_count,
                "n_valid_annotations": n_ann,
            }
            for label in ["A", "B"]:
                if f"llm_{label}_{s}" in r:
                    sr[f"llm_prob_{label}"] = r[f"llm_{label}_{s}"]
            sense_rows.append(sr)
    return sense_rows


def compute_risk_quadrants(sense_rows: list[dict], variant: str) -> tuple[float, float]:
    """Label each (item, sense) row with its risk quadrant for the given variant.

    Crowd axis is binned on the RAW annotator vote count: a sense with only
    one annotator vote is "low" (singleton, likely noise/ambiguity); two or
    more votes is "high". This avoids a float-precision artifact where 1/9,
    1/10 and 1/11 (all "one annotator picked this sense") land on different
    sides of a probability quartile cutoff and get different quadrant labels
    despite being visually indistinguishable on the plot.

    LLM axis stays continuous: cutoff is the 25th percentile of LLM
    probability over all (item, sense) points; "low" = at or below the cutoff.

    Returns (crowd_cut, llm_cut) where crowd_cut is the count threshold (1.0)
    for backward compatibility with the existing plot/CSV code.
    """
    key = f"llm_prob_{variant}"
    sub = [r for r in sense_rows if key in r]
    if not sub:
        return float("nan"), float("nan")

    crowd_cut = 1.0  # vote count threshold: low <=> count == 1
    llm_cut = float(np.percentile([r[key] for r in sub], 25))

    for r in sub:
        crowd_bin = "low" if r["crowd_count"] <= 1 else "high"
        llm_bin = "low" if r[key] <= llm_cut else "high"
        r[f"quadrant_{variant}"] = QUADRANT_LABELS[(crowd_bin, llm_bin)]
    return crowd_cut, llm_cut


def plot_risk_quadrants(sense_rows: list[dict], cutoffs: dict[str, tuple[float, float]],
                        out_dir: Path):
    """E1: 2x2 risk quadrant scatter per variant, one point per (item, sense)."""
    variants = [v for v in ["A", "B"] if any(f"quadrant_{v}" in r for r in sense_rows)]
    if not variants:
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(8 * len(variants), 6.5),
                             squeeze=False)
    sense_markers = {"temporal": "o", "contingency": "s",
                     "comparison": "^", "expansion": "D"}

    for ax, variant in zip(axes[0], variants):
        crowd_cut, llm_cut = cutoffs[variant]
        sub = [r for r in sense_rows if f"quadrant_{variant}" in r]

        for quad, color in QUADRANT_COLORS.items():
            qr = [r for r in sub if r[f"quadrant_{variant}"] == quad]
            for sense, marker in sense_markers.items():
                sr = [r for r in qr if r["sense"] == sense]
                if not sr:
                    continue
                ax.scatter([r["crowd_prob"] for r in sr],
                           [r[f"llm_prob_{variant}"] for r in sr],
                           c=color, marker=marker, s=45, alpha=0.7,
                           edgecolors="none")
            if qr:
                ax.scatter([], [], c=color, label=f"{quad} (n={len(qr)})", s=45)

        # Crowd cutoff sits between "1 vote" and "2 votes": draw it at the
        # midpoint of those two values in the actual probability data so the
        # dashed line lies between the two visually distinct strips.
        singles = [r["crowd_prob"] for r in sub if r["crowd_count"] == 1]
        multis = [r["crowd_prob"] for r in sub if r["crowd_count"] >= 2]
        x_cut = (max(singles) + min(multis)) / 2 if singles and multis else 0.105
        ax.axvline(x_cut, ls="--", color="gray", alpha=0.7)
        ax.axhline(llm_cut, ls="--", color="gray", alpha=0.7)
        ax.set_xlabel("Crowd Probability per Sense (low = singleton vote)")
        ax.set_ylabel(f"LLM Probability per Sense (Q25 cutoff = {llm_cut:.2f})")
        n = len(sub)
        ax.set_title(f"Risk Quadrants — Variante {variant} ({n} item-sense points)")
        handles, labels = ax.get_legend_handles_labels()
        for sense, marker in sense_markers.items():
            handles.append(ax.scatter([], [], c="gray", marker=marker, s=45))
            labels.append(sense)
        ax.legend(handles, labels, loc="upper left", framealpha=0.9, ncol=2)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_dir / "risk_quadrants.png")
    plt.close()


def write_risk_quadrants_csv(sense_rows: list[dict],
                             cutoffs: dict[str, tuple[float, float]],
                             out_dir: Path):
    """E2: Per-(item, sense) quadrant table, high-risk points first."""
    import csv
    cols = ["item_id", "sense", "genre", "crowd_agreement", "crowd_majority",
            "n_valid_annotations", "crowd_count", "crowd_prob",
            "llm_prob_A", "llm_prob_B",
            "quadrant_A", "quadrant_B"]
    quad_order = {"high_risk": 0, "llm_overconfident": 1,
                  "llm_underconfident": 2, "safe": 3}

    labeled = [r for r in sense_rows if "quadrant_A" in r or "quadrant_B" in r]
    labeled.sort(key=lambda r: (quad_order.get(r.get("quadrant_B", r.get("quadrant_A")), 9),
                                r["crowd_prob"]))

    path = out_dir / "risk_quadrants.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in labeled:
            w.writerow(r)
    print(f"  Wrote {path} ({len(labeled)} item-sense rows)")

    for variant, (_, llm_cut) in cutoffs.items():
        counts = Counter(r[f"quadrant_{variant}"] for r in sense_rows
                         if f"quadrant_{variant}" in r)
        print(f"  Variante {variant}: cutoffs crowd_count==1 (singleton), "
              f"llm_prob<={llm_cut:.2f} | "
              + ", ".join(f"{q}={counts.get(q, 0)}" for q in quad_order))


# ---------------------------------------------------------------------------
# F: Wikipedia gold labels (DiscoGeM partial gold)
# ---------------------------------------------------------------------------
# The reflabel column uses fine-grained PDTB-3 sense tokens, while our
# analysis axis is PDTB Level-1 (temporal/contingency/comparison/expansion).
# This mapping follows the standard PDTB-3 hierarchy. Tokens not listed are
# dropped (and reported as unknown so missing senses are visible).
PDTB_REFLABEL_TO_L1 = {
    # Temporal
    "synchronous": "temporal", "asynchronous": "temporal",
    "precedence": "temporal", "succession": "temporal",
    # Contingency
    "reason": "contingency", "result": "contingency",
    "arg1-as-cond": "contingency", "arg2-as-cond": "contingency",
    "arg1-as-negcond": "contingency", "arg2-as-negcond": "contingency",
    "arg1-as-goal": "contingency", "arg2-as-goal": "contingency",
    "arg2-as-purpose": "contingency",
    # Comparison
    "contrast": "comparison", "similarity": "comparison",
    "concession": "comparison",
    "arg1-as-denier": "comparison", "arg2-as-denier": "comparison",
    # Expansion
    "conjunction": "expansion", "disjunction": "expansion",
    "equivalence": "expansion",
    "arg1-as-detail": "expansion", "arg2-as-detail": "expansion",
    "arg1-as-instance": "expansion", "arg2-as-instance": "expansion",
    "arg1-as-subst": "expansion", "arg2-as-subst": "expansion",
    "arg1-as-excpt": "expansion", "arg2-as-excpt": "expansion",
    "arg1-as-manner": "expansion", "arg2-as-manner": "expansion",
}


def load_wiki_gold_l1(project_root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Return ({item_id: set_of_L1_senses}, set_of_unknown_tokens) from reflabels.

    Reflabels are semicolon-separated PDTB-3 sense tokens. Tokens are mapped
    via PDTB_REFLABEL_TO_L1; "norel" is dropped; unknown tokens are collected
    in the second return value so we can warn about them.
    """
    import csv as _csv
    wide = project_root / "DiscoGeM 1.0_items" / "DiscoGeM1.0.wide.csv"
    if not wide.exists():
        return {}, set()
    gold: dict[str, set[str]] = {}
    unknown: set[str] = set()
    with wide.open(newline="") as f:
        for row in _csv.DictReader(f):
            ref = (row.get("reflabel") or "").strip()
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
                else:
                    unknown.add(tok)
            if l1_set:
                gold[str(row["itemid"])] = l1_set
    return gold, unknown


def annotate_wiki_gold(sense_rows: list[dict], gold_map: dict[str, set[str]]) -> None:
    """Mark sense_rows whose (item, sense) is a Wikipedia gold reference."""
    for r in sense_rows:
        item_gold = gold_map.get(r["item_id"], set())
        r["is_wiki_gold"] = r["genre"] == "Wiki" and r["sense"] in item_gold
        r["gold_l1_senses"] = ";".join(sorted(item_gold)) if item_gold else ""


def plot_wiki_gold_quadrants(sense_rows: list[dict],
                             cutoffs: dict[str, tuple[float, float]],
                             out_dir: Path):
    """F1: 2x2 risk-quadrant scatter with Wikipedia gold points highlighted."""
    variants = [v for v in ["A", "B"] if v in cutoffs]
    if not variants:
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(8 * len(variants), 6.5),
                             squeeze=False)

    for ax, variant in zip(axes[0], variants):
        crowd_cut, llm_cut = cutoffs[variant]
        key = f"llm_prob_{variant}"
        sub = [r for r in sense_rows if f"quadrant_{variant}" in r]

        # Non-Wiki background
        bg = [r for r in sub if r["genre"] != "Wiki"]
        ax.scatter([r["crowd_prob"] for r in bg], [r[key] for r in bg],
                   c="lightgray", s=30, alpha=0.5, edgecolors="none",
                   label=f"non-Wiki (n={len(bg)})")

        # Wiki non-gold
        wiki_other = [r for r in sub if r["genre"] == "Wiki" and not r["is_wiki_gold"]]
        for quad, color in QUADRANT_COLORS.items():
            qr = [r for r in wiki_other if r[f"quadrant_{variant}"] == quad]
            if qr:
                ax.scatter([r["crowd_prob"] for r in qr],
                           [r[key] for r in qr],
                           c=color, s=55, alpha=0.5, edgecolors="none")

        # Wiki gold highlighted
        gold = [r for r in sub if r["is_wiki_gold"]]
        gold_counts = Counter(r[f"quadrant_{variant}"] for r in gold)
        for quad, color in QUADRANT_COLORS.items():
            qr = [r for r in gold if r[f"quadrant_{variant}"] == quad]
            if not qr:
                continue
            ax.scatter([r["crowd_prob"] for r in qr],
                       [r[key] for r in qr],
                       c=color, s=140, alpha=0.95,
                       edgecolors="black", linewidths=1.5,
                       label=f"gold {quad} (n={len(qr)})")
            for r in qr:
                ax.annotate(r["item_id"].replace("wiki_", "w"),
                            (r["crowd_prob"], r[key]),
                            fontsize=7, alpha=0.85,
                            xytext=(5, 5), textcoords="offset points")

        singles = [r["crowd_prob"] for r in sub if r.get("crowd_count") == 1]
        multis = [r["crowd_prob"] for r in sub if r.get("crowd_count", 0) >= 2]
        x_cut = (max(singles) + min(multis)) / 2 if singles and multis else 0.105
        ax.axvline(x_cut, ls="--", color="gray", alpha=0.7)
        ax.axhline(llm_cut, ls="--", color="gray", alpha=0.7)
        ax.set_xlabel("Crowd Probability per Sense (low = singleton vote)")
        ax.set_ylabel(f"LLM Probability per Sense (Q25 = {llm_cut:.2f})")
        n_gold = sum(gold_counts.values())
        ax.set_title(f"Wikipedia Gold Labels — Variante {variant} ({n_gold} gold points)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_dir / "risk_quadrants_wiki_gold.png")
    plt.close()


def write_wiki_gold_stats(sense_rows: list[dict],
                          rows: list[dict],
                          gold_map: dict[str, set[str]],
                          cutoffs: dict[str, tuple[float, float]],
                          out_dir: Path):
    """F2: Per-gold-point CSV + per-variant JSON stats and a console table.

    Also reports gold senses that the crowd never picked (filtered out of the
    quadrant analysis) - these are editor-vs-crowd disagreements where the
    LLM's stance matters most.
    """
    import csv as _csv
    gold = [r for r in sense_rows if r.get("is_wiki_gold")]

    # Find gold (item, sense) pairs that were filtered out (crowd_prob == 0).
    sense_rows_index = {(r["item_id"], r["sense"]) for r in sense_rows}
    rows_by_id = {r["item_id"]: r for r in rows}
    no_crowd_gold = []
    for iid, l1_set in gold_map.items():
        if iid not in rows_by_id or rows_by_id[iid].get("genre") != "Wiki":
            continue
        for s in l1_set:
            if (iid, s) not in sense_rows_index:
                pr = rows_by_id[iid]
                no_crowd_gold.append({
                    "item_id": iid, "sense": s,
                    "crowd_prob": pr.get(f"crowd_{s}", 0.0),
                    "llm_prob_A": pr.get(f"llm_A_{s}"),
                    "llm_prob_B": pr.get(f"llm_B_{s}"),
                    "gold_l1_senses": ";".join(sorted(l1_set)),
                })

    if not gold and not no_crowd_gold:
        print("  No Wikipedia gold labels found - skipping stats.")
        return

    cols = ["item_id", "sense", "crowd_prob", "llm_prob_A", "llm_prob_B",
            "quadrant_A", "quadrant_B", "gold_l1_senses"]
    quad_order = ["high_risk", "llm_overconfident", "llm_underconfident", "safe"]
    path_csv = out_dir / "wiki_gold_quadrants.csv"
    gold_sorted = sorted(gold, key=lambda r: (r["item_id"], r["sense"]))
    with path_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in gold_sorted:
            w.writerow(r)
    print(f"  Wrote {path_csv} ({len(gold)} gold item-sense rows)")

    if no_crowd_gold:
        path_nc = out_dir / "wiki_gold_no_crowd_support.csv"
        nc_cols = ["item_id", "sense", "crowd_prob", "llm_prob_A", "llm_prob_B",
                   "gold_l1_senses"]
        with path_nc.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=nc_cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(no_crowd_gold, key=lambda r: (r["item_id"], r["sense"])):
                w.writerow(r)
        print(f"  Wrote {path_nc} ({len(no_crowd_gold)} editor-vs-crowd conflicts)")

    stats: dict = {"n_no_crowd_support_gold_points": len(no_crowd_gold)}
    for variant in cutoffs:
        qkey = f"quadrant_{variant}"
        counts = Counter(r[qkey] for r in gold if qkey in r)
        # "fully confirmed" = every gold sense of an item lands in safe
        items_with_gold = sorted({r["item_id"] for r in gold})
        items_all_safe = []
        items_with_concerning = []
        for iid in items_with_gold:
            quads = [r[qkey] for r in gold if r["item_id"] == iid and qkey in r]
            if quads and all(q == "safe" for q in quads):
                items_all_safe.append(iid)
            if any(q in ("high_risk", "llm_underconfident") for q in quads):
                items_with_concerning.append(iid)
        n_gold = sum(counts.values())
        stats[f"variante_{variant}"] = {
            "n_gold_points": n_gold,
            "counts_by_quadrant": {q: counts.get(q, 0) for q in quad_order},
            "share_safe": counts.get("safe", 0) / n_gold if n_gold else 0.0,
            "n_items_with_gold": len(items_with_gold),
            "n_items_all_gold_safe": len(items_all_safe),
            "items_with_concerning_gold": items_with_concerning,
        }

    path_json = out_dir / "wiki_gold_stats.json"
    with path_json.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote {path_json}")

    for variant in cutoffs:
        s = stats[f"variante_{variant}"]
        print(f"  Variante {variant}: {s['n_gold_points']} gold points | "
              + ", ".join(f"{q}={s['counts_by_quadrant'][q]}" for q in quad_order)
              + f" | safe share={s['share_safe']:.0%} | "
              + f"{s['n_items_all_gold_safe']}/{s['n_items_with_gold']} items fully safe")
        if s["items_with_concerning_gold"]:
            print(f"    concerning items: {', '.join(s['items_with_concerning_gold'])}")
    if no_crowd_gold:
        print(f"  {len(no_crowd_gold)} editor-vs-crowd conflicts "
              "(gold sense the crowd never picked, excluded from quadrants):")
        for r in sorted(no_crowd_gold, key=lambda r: r["item_id"]):
            la = f"{r['llm_prob_A']:.2f}" if r['llm_prob_A'] is not None else "-"
            lb = f"{r['llm_prob_B']:.2f}" if r['llm_prob_B'] is not None else "-"
            print(f"    {r['item_id']:12s} sense={r['sense']:11s} "
                  f"crowd=0  llm_A={la}  llm_B={lb}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_summary_csv(rows: list[dict], out_dir: Path):
    """A6: Per-item summary table."""
    import csv
    cols = ["item_id", "genre", "stratification_bin", "crowd_agreement", "crowd_majority",
            "n_crowd_senses", "llm_top1_A", "llm_top1_B", "top1_match_A", "top1_match_B",
            "jsd_A", "jsd_B", "kld_A", "kld_B", "mae_A", "mae_B", "rmse_A", "rmse_B"]
    path = out_dir / "summary_metrics.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r.get("jsd_B", 0)):
            w.writerow(r)
    print(f"  Wrote {path}")


def write_aggregate_json(rows: list[dict], out_dir: Path):
    """A7: Global aggregate metrics."""
    result = {}
    for label in ["A", "B"]:
        jsd_vals = [r[f"jsd_{label}"] for r in rows if f"jsd_{label}" in r]
        kld_vals = [r[f"kld_{label}"] for r in rows if f"kld_{label}" in r]
        mae_vals = [r[f"mae_{label}"] for r in rows if f"mae_{label}" in r]
        rmse_vals = [r[f"rmse_{label}"] for r in rows if f"rmse_{label}" in r]
        top1 = [r[f"top1_match_{label}"] for r in rows if f"top1_match_{label}" in r]

        all_cx = [r[f"crowd_{s}"] for r in rows for s in SENSES if f"llm_{label}_{s}" in r]
        all_ly = [r[f"llm_{label}_{s}"] for r in rows for s in SENSES if f"llm_{label}_{s}" in r]

        result[f"variante_{label}"] = {
            "n_items": len(jsd_vals),
            "jsd_mean": mean(jsd_vals) if jsd_vals else 0,
            "jsd_std": stdev(jsd_vals) if len(jsd_vals) > 1 else 0,
            "jsd_median": sorted(jsd_vals)[len(jsd_vals) // 2] if jsd_vals else 0,
            "kld_mean": mean(kld_vals) if kld_vals else 0,
            "mae_mean": mean(mae_vals) if mae_vals else 0,
            "rmse_mean": mean(rmse_vals) if rmse_vals else 0,
            "top1_accuracy": sum(top1) / len(top1) if top1 else 0,
            "pearson_r": pearson_r(all_cx, all_ly),
            "spearman_rho": spearman_rho(all_cx, all_ly),
        }

    path = out_dir / "aggregate_metrics.json"
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {path}")


def write_genre_breakdown_csv(rows: list[dict], out_dir: Path):
    """B5: Genre x Sense breakdown table."""
    import csv
    path = out_dir / "genre_sense_breakdown.csv"
    cols = ["genre", "sense", "n_items", "mean_crowd_prob", "mean_llm_prob_B", "bias_B", "mae_B"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for genre in ["Europarl", "Lit", "Wiki"]:
            gr = [r for r in rows if r["genre"] == genre]
            for s in SENSES:
                crowd_vals = [r[f"crowd_{s}"] for r in gr]
                llm_vals = [r[f"llm_B_{s}"] for r in gr if f"llm_B_{s}" in r]
                mc = mean(crowd_vals) if crowd_vals else 0
                ml = mean(llm_vals) if llm_vals else 0
                w.writerow({
                    "genre": genre,
                    "sense": s,
                    "n_items": len(gr),
                    "mean_crowd_prob": f"{mc:.4f}",
                    "mean_llm_prob_B": f"{ml:.4f}",
                    "bias_B": f"{ml - mc:+.4f}",
                    "mae_B": f"{mean(abs(c - l) for c, l in zip(crowd_vals, llm_vals)):.4f}" if llm_vals else "0",
                })
    print(f"  Wrote {path}")


def write_model_comparison_csv(
    items: list[PrepItem],
    comp_results: list[dict],
    gens: list[GenerationRecord],
    vals: list[ValidationRecord],
    out_dir: Path,
):
    """C6: Per-model summary table."""
    import csv
    items_by_id = {it.item_id: it for it in items}
    models = sorted(set(r["model"] for r in comp_results))

    # Per-model JSD
    model_jsds = {m: [] for m in models}
    model_top1 = {m: [] for m in models}
    for r in comp_results:
        item = items_by_id.get(r["item_id"])
        if not item:
            continue
        crowd = _normalize_crowd(item.crowd_sense_distribution)
        llm = {s: r["distribution"].get(s, 0.0) for s in SENSES}
        model_jsds[r["model"]].append(jsd(crowd, llm))
        crowd_top = max(crowd, key=crowd.get)
        llm_top = max(llm, key=llm.get)
        model_top1[r["model"]].append(crowd_top == llm_top)

    # Validator/generator stats
    val_scores = defaultdict(list)
    for v in vals:
        val_scores[v.validator_model].append(v.validity_score)

    gen_id_to_model = {g.generation_id: g.generator_model for g in gens}
    gen_scores = defaultdict(list)
    for v in vals:
        gm = gen_id_to_model.get(v.generation_id)
        if gm:
            gen_scores[gm].append(v.validity_score)

    abstentions = Counter(g.generator_model for g in gens if g.abstained)
    total_gens = Counter(g.generator_model for g in gens)

    path = out_dir / "model_comparison.csv"
    cols = ["model", "model_short", "mean_jsd", "top1_accuracy",
            "mean_val_given", "mean_val_received", "abstention_rate", "n_abstentions"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in models:
            w.writerow({
                "model": m,
                "model_short": MODEL_SHORT.get(m, m),
                "mean_jsd": f"{mean(model_jsds[m]):.4f}" if model_jsds[m] else "",
                "top1_accuracy": f"{sum(model_top1[m]) / len(model_top1[m]):.4f}" if model_top1[m] else "",
                "mean_val_given": f"{mean(val_scores[m]):.4f}" if val_scores[m] else "",
                "mean_val_received": f"{mean(gen_scores[m]):.4f}" if gen_scores[m] else "",
                "abstention_rate": f"{abstentions[m] / total_gens[m]:.4f}" if total_gens[m] else "0",
                "n_abstentions": abstentions[m],
            })
    print(f"  Wrote {path}")


def write_fleiss_kappa_json(comp_results: list[dict], items: list[PrepItem], out_dir: Path):
    """C7: Fleiss' kappa."""
    kappa = compute_fleiss_kappa(comp_results, items)
    path = out_dir / "fleiss_kappa.json"
    with path.open("w") as f:
        json.dump({"fleiss_kappa_top1": kappa, "n_items": len(items), "n_raters": 4}, f, indent=2)
    print(f"  Wrote {path} (κ = {kappa:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis of EVADE pipeline results")
    parser.add_argument("--run-dir", required=True, help="Path to results directory")
    parser.add_argument("--tau", default="0.30", help="Tau for Variante A (default: 0.30)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis_detailed"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading data from {run_dir}...")
    items = read_items(run_dir / "items.jsonl")
    gens = read_generations(run_dir / "generations.jsonl")
    vals = read_validations(run_dir / "validations.jsonl")

    dists_a_path = run_dir / "distributions_aggregate.jsonl"
    dists_b_path = run_dir / "distributions.jsonl"
    dists_a = read_distributions(dists_a_path) if dists_a_path.exists() else []
    dists_b = read_distributions(dists_b_path) if dists_b_path.exists() else []

    comp_path = run_dir / "comparative_results.jsonl"
    comp_results = load_comparative_results(comp_path) if comp_path.exists() else []

    print(f"  {len(items)} items, {len(gens)} generations, {len(vals)} validations")
    print(f"  {len(dists_a)} distributions (A), {len(dists_b)} distributions (B)")
    print(f"  {len(comp_results)} comparative results")

    # Compute per-item metrics
    print(f"\nComputing metrics (tau={args.tau})...")
    rows = compute_per_item_metrics(items, dists_a, dists_b, tau=args.tau)

    # A: LLM vs Crowd
    print("\n[A] LLM vs Crowd...")
    plot_jsd_per_item(rows, out_dir)
    plot_correlation_scatter(rows, out_dir)
    if dists_b:
        plot_crowd_vs_llm_covariance(items, dists_b, out_dir)
    plot_calibration(rows, out_dir)
    plot_per_sense_bias(rows, out_dir)
    plot_confusion_matrix(rows, out_dir)
    write_summary_csv(rows, out_dir)
    write_aggregate_json(rows, out_dir)

    # B: Between categories
    print("[B] Between Categories...")
    plot_genre_boxplot(rows, out_dir)
    plot_sense_genre_heatmap(rows, out_dir)
    plot_confusion_by_genre(rows, out_dir)
    plot_agreement_vs_jsd(rows, out_dir)
    write_genre_breakdown_csv(rows, out_dir)

    # C: Between LLMs
    print("[C] Between LLMs...")
    if comp_results:
        plot_model_distributions(comp_results, out_dir)
        plot_model_jsd_boxplot(items, comp_results, out_dir)
        plot_inter_model_agreement(comp_results, out_dir)
        write_model_comparison_csv(items, comp_results, gens, vals, out_dir)
        write_fleiss_kappa_json(comp_results, items, out_dir)
    if gens and vals:
        plot_validator_generator_quality(gens, vals, out_dir)
        plot_abstention_heatmap(gens, out_dir)

    # D: Variante comparison
    print("[D] Variante A vs B...")
    if dists_a and dists_b:
        plot_variante_scatter(rows, out_dir)
        plot_variante_comparison(rows, out_dir)

    # E: Risk quadrants (per item-sense point)
    print("[E] Risk quadrants...")
    sense_rows = build_sense_rows(rows)
    cutoffs = {}
    for variant in ["A", "B"]:
        crowd_cut, llm_cut = compute_risk_quadrants(sense_rows, variant)
        if not math.isnan(crowd_cut):
            cutoffs[variant] = (crowd_cut, llm_cut)
    if cutoffs:
        plot_risk_quadrants(sense_rows, cutoffs, out_dir)
        write_risk_quadrants_csv(sense_rows, cutoffs, out_dir)

    # F: Wikipedia gold labels
    print("[F] Wikipedia gold labels...")
    gold_map, unknown_tokens = load_wiki_gold_l1(PROJECT_ROOT)
    annotate_wiki_gold(sense_rows, gold_map)
    run_item_ids = {r["item_id"] for r in rows}
    n_wiki_items_with_gold = sum(1 for iid in gold_map if iid in run_item_ids)
    print(f"  Loaded {len(gold_map)} items with gold reflabel "
          f"({n_wiki_items_with_gold} present in this run)")
    if unknown_tokens:
        print(f"  Unknown reflabel tokens (no L1 mapping): {sorted(unknown_tokens)}")
    if cutoffs and any(r.get("is_wiki_gold") for r in sense_rows):
        plot_wiki_gold_quadrants(sense_rows, cutoffs, out_dir)
        write_wiki_gold_stats(sense_rows, rows, gold_map, cutoffs, out_dir)

    print(f"\nDone. All artifacts in {out_dir}")


if __name__ == "__main__":
    main()
