# Fine-Tuning Experiment

Adapts the supervisor's NLI fine-tuning experiment (Setup a) to the
DiscoGeM Level-1 discourse-relation task, with one twist that ties
directly to our methodological contribution: a third training-label
variant derived from the **risk-quadrant analysis**.

## Hypothesis

> Fine-tuning a small transformer on **quadrant-curated** label
> distributions yields better alignment with Wikipedia editor-gold than
> either raw crowd labels or raw EVADE labels.

The Wikipedia gold acts as a genuinely external evaluation signal — it
is *never* used during training and never enters the risk-quadrant
computation, so it cannot leak into the curation rule.

## Pipeline

1. **`prepare_data.py`** — Builds three training sets and a test set
   from the Wiki-extension run directory:

   - `train_raw_crowd.jsonl` — target = crowd-derived L1 distribution
   - `train_raw_evade.jsonl` — target = our `distributions.jsonl`
     Variante B (LLM-derived distribution at τ=0.30)
   - `train_quadrant_curated.jsonl` — target = crowd distribution with
     senses marked `llm_overconfident` or `high_risk` zeroed out and
     renormalised
   - `test.jsonl` — 20% held-out items, each annotated with both the
     crowd distribution and the Wikipedia editor-gold L1 senses

2. **`train.py`** — Fine-tunes a BERT- or RoBERTa-style backbone with
   a single dense classification head on top of `[CLS]`. Loss is
   KL-divergence between the predicted softmax over 4 senses and the
   training target. Auto-detects MPS (Apple Silicon), CUDA, or CPU.

3. **`evaluate.py`** — For each test item, predicts the L1
   distribution and reports

   - KL and JSD vs the crowd distribution
   - KL and JSD vs the Wikipedia gold distribution
   - Top-1 prediction accuracy (against crowd majority and against
     gold set)
   - Multi-label F1 (predicted senses ≥ 0.30 vs gold set)

4. **`summarize.py`** — Aggregates the six summary JSONs (2 backbones
   × 3 variants) into a comparison table (`summary_table.csv` /
   `summary_table.md`) and a side-by-side bar plot
   (`summary_plot.png`).

## Curation Rule

For each (item, sense) row in
`analysis_detailed/risk_quadrants.csv`:

| Quadrant (Variante B) | Action on the crowd probability |
|---|---|
| `safe` | Keep as is |
| `llm_underconfident` | Keep as is — crowd is reliable |
| `llm_overconfident` | **Set to 0** — likely commission error |
| `high_risk` | **Set to 0** — noisy point |

After zeroing, the per-item distribution is renormalised to sum to 1.
If every sense was zeroed (rare), we fall back to the raw crowd
distribution.

In the current Wiki-extension run, **279 of 504 training items (55%)
get their distribution modified by this rule** — non-trivial signal,
not just cosmetic.

## How to Run

Requires the Wiki-extension run completed (the 630-item
`distributions.jsonl` and `analysis_detailed/risk_quadrants.csv`).

```bash
# Install deps (one-time)
~/.pyenv/versions/3.11.7/bin/pip install torch transformers

# Full pipeline: 6 trainings + 6 evals + summary
bash scripts/finetune/run_all.sh
```

Override defaults via env vars:

```bash
RUN_DIR=results/20260625T062938Z_wiki_extension_bcb6af9 \
EPOCHS=10 BATCH_SIZE=16 \
bash scripts/finetune/run_all.sh
```

Outputs go to `results/finetune/`:

```
results/finetune/
├── data/                     # 4 JSONL files
├── models/
│   ├── bert-base-uncased_raw_crowd/
│   ├── bert-base-uncased_raw_evade/
│   ├── bert-base-uncased_quadrant_curated/
│   ├── roberta-base_raw_crowd/
│   ├── roberta-base_raw_evade/
│   └── roberta-base_quadrant_curated/
├── eval/
│   ├── eval_<backbone>_<variant>.csv
│   └── eval_<backbone>_<variant>.summary.json
└── summary/
    ├── summary_table.csv
    ├── summary_table.md
    └── summary_plot.png
```

## Compute & Cost

| Stage | Mac M-series (MPS) | NVIDIA A100 |
|---|---|---|
| Per training (BERT-base, 5 epochs, 504 items) | ~5–10 min | ~1–2 min |
| Full sweep (6 trainings + evals) | ~45–60 min | ~15 min |

Disk: ~3 GB total (six 440 MB models + datasets). No API costs — all
local.

## Expected Outcome

If the hypothesis holds:

| | KL vs Gold ↓ | F1 vs Gold ↑ |
|---|---|---|
| BERT/raw_crowd | baseline | baseline |
| BERT/raw_evade | similar | similar |
| **BERT/quadrant_curated** | **lower** | **higher** |

A monotone improvement from `raw_crowd` → `raw_evade` →
`quadrant_curated` is the strongest result the experiment can
produce. Even an improvement of `quadrant_curated` over both baselines
on just the gold metric is publishable.

If the curated variant is **worse**, the interpretation is still
interesting: it would mean the quadrants are too aggressive a filter,
or that the wiki crowd is already cleaner than we thought.

## Caveats

- N=504 train / 126 test is small for transformer fine-tuning. Loss
  variance across seeds may matter; consider reporting mean ± std over
  3 seeds for the paper version.
- No pre-training step on the rest of DiscoGeM (the supervisor used
  MNLI pre-training). Easy to add later — would likely improve all
  three variants uniformly without changing the *comparison*.
- The threshold `pred ≥ 0.30` for multi-label F1 is a fixed
  hyperparameter. Sweeping it would give a Precision-Recall curve;
  for the paper a single threshold + the curve in the appendix is
  sufficient.
