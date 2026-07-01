#!/usr/bin/env bash
# End-to-end fine-tuning experiment, multi-seed.
#   1. Build train/test data (single shared split across seeds)
#   2. For each (backbone, label_variant, seed): train + evaluate
#   3. Aggregate seed-wise results into mean ± std summary
#
# Output lands in results/finetune/.
# Existing seed runs are skipped (model.pt already present).

set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-$HOME/.pyenv/versions/3.11.7/bin/python}"
RUN_DIR="${RUN_DIR:-results/20260625T062938Z_wiki_extension_bcb6af9}"
OUT_DIR="${OUT_DIR:-results/finetune}"
DATA_DIR="$OUT_DIR/data"
MODELS_DIR="$OUT_DIR/models"
EVAL_DIR="$OUT_DIR/eval"
SUMMARY_DIR="$OUT_DIR/summary"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-5}"

BACKBONES=( "bert-base-uncased" "roberta-base" )
VARIANTS=( "raw_crowd" "raw_evade" "quadrant_curated" )
SEEDS=( "42" "123" "2024" )

echo "[run_all] python: $PY"
echo "[run_all] run dir: $RUN_DIR"
echo "[run_all] output: $OUT_DIR"
echo "[run_all] backbones: ${BACKBONES[*]}"
echo "[run_all] variants:  ${VARIANTS[*]}"
echo "[run_all] seeds:     ${SEEDS[*]}"

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$EVAL_DIR" "$SUMMARY_DIR"

# 1. Build data (only if test.jsonl missing)
if [[ ! -f "$DATA_DIR/test.jsonl" ]]; then
  echo ""
  echo "=== [1] Preparing data ==="
  "$PY" scripts/finetune/prepare_data.py \
    --run-dir "$RUN_DIR" \
    --out-dir "$DATA_DIR"
else
  echo "[run_all] data already present, skipping prepare"
fi

# 2 + 3. Train + evaluate each combination
i=0
total=$(( ${#BACKBONES[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]} ))
for bb in "${BACKBONES[@]}"; do
  bb_safe=$(echo "$bb" | tr '/' '_')
  for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      i=$((i + 1))
      label="${bb_safe}_${variant}_seed${seed}"
      model_out="$MODELS_DIR/$label"
      eval_csv="$EVAL_DIR/eval_${label}.csv"
      echo ""
      echo "=== [$i/$total] $bb / $variant / seed=$seed ==="
      if [[ -f "$model_out/model.pt" ]]; then
        echo "[run_all] model exists, skipping training"
      else
        "$PY" scripts/finetune/train.py \
          --train-jsonl "$DATA_DIR/train_${variant}.jsonl" \
          --backbone "$bb" \
          --out-dir "$model_out" \
          --epochs "$EPOCHS" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --seed "$seed"
      fi
      if [[ -f "$eval_csv" ]] && [[ -f "${eval_csv%.csv}.summary.json" ]]; then
        echo "[run_all] eval exists, skipping"
      else
        "$PY" scripts/finetune/evaluate.py \
          --model-dir "$model_out" \
          --test-jsonl "$DATA_DIR/test.jsonl" \
          --out-csv "$eval_csv"
      fi
    done
  done
done

# 4. Summarize across seeds
echo ""
echo "=== [4] Summarizing across seeds ==="
"$PY" scripts/finetune/summarize.py \
  --results-dir "$EVAL_DIR" \
  --out-dir "$SUMMARY_DIR"

echo ""
echo "Done. Results in $OUT_DIR"
