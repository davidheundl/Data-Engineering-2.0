#!/usr/bin/env bash
# Pre-train + multi-seed fine-tune.
#   1. Build pretrain.jsonl from non-Wiki-extension DiscoGeM items
#   2. For each backbone: pre-train on pretrain.jsonl
#   3. For each (backbone, variant, seed): fine-tune from pre-trained checkpoint
#   4. Aggregate into a comparison table
#
# Fine-tuned models have suffix `_pt` to distinguish them from the non-
# pre-trained runs already on disk. summarize.py picks both up.

set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-$HOME/.pyenv/versions/3.11.7/bin/python}"
RUN_DIR="${RUN_DIR:-results/20260625T062938Z_wiki_extension_bcb6af9}"
OUT_DIR="${OUT_DIR:-results/finetune}"
DATA_DIR="$OUT_DIR/data"
PRETRAIN_DIR="$OUT_DIR/pretrained"
MODELS_DIR="$OUT_DIR/models"
EVAL_DIR="$OUT_DIR/eval"
SUMMARY_DIR="$OUT_DIR/summary"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
PRETRAIN_LR="${PRETRAIN_LR:-2e-5}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-5}"

BACKBONES=( "bert-base-uncased" "roberta-base" )
VARIANTS=( "raw_crowd" "raw_evade" "quadrant_curated" )
SEEDS=( "42" "123" "2024" )

echo "[run] python: $PY"
echo "[run] backbones: ${BACKBONES[*]}"
echo "[run] variants:  ${VARIANTS[*]}"
echo "[run] seeds:     ${SEEDS[*]}"

mkdir -p "$DATA_DIR" "$PRETRAIN_DIR" "$MODELS_DIR" "$EVAL_DIR" "$SUMMARY_DIR"

# 1. Build pretrain.jsonl (if missing)
if [[ ! -f "$DATA_DIR/pretrain.jsonl" ]]; then
  echo ""
  echo "=== [1] Building pretrain.jsonl ==="
  "$PY" scripts/finetune/pretrain_data.py \
    --exclude-run-dir "$RUN_DIR" \
    --out "$DATA_DIR/pretrain.jsonl"
else
  echo "[run] pretrain.jsonl already present"
fi

# 2. Pre-train each backbone
for bb in "${BACKBONES[@]}"; do
  bb_safe=$(echo "$bb" | tr '/' '_')
  pt_dir="$PRETRAIN_DIR/$bb_safe"
  echo ""
  echo "=== [2/${#BACKBONES[@]}] Pre-training $bb on ~5800 DiscoGeM items ==="
  if [[ -f "$pt_dir/model.pt" ]]; then
    echo "[run] pretrained checkpoint exists at $pt_dir, skipping"
  else
    "$PY" scripts/finetune/train.py \
      --train-jsonl "$DATA_DIR/pretrain.jsonl" \
      --backbone "$bb" \
      --out-dir "$pt_dir" \
      --epochs "$PRETRAIN_EPOCHS" \
      --batch-size "$PRETRAIN_BATCH_SIZE" \
      --lr "$PRETRAIN_LR" \
      --seed 42
  fi
done

# 3 + 4. Fine-tune each (backbone, variant, seed) combo from the pre-trained checkpoint
i=0
total=$(( ${#BACKBONES[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]} ))
for bb in "${BACKBONES[@]}"; do
  bb_safe=$(echo "$bb" | tr '/' '_')
  pt_ckpt="$PRETRAIN_DIR/$bb_safe/model.pt"
  for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      i=$((i + 1))
      label="${bb_safe}_${variant}_seed${seed}_pt"
      model_out="$MODELS_DIR/$label"
      eval_csv="$EVAL_DIR/eval_${label}.csv"
      echo ""
      echo "=== [$i/$total] PT-FT $bb / $variant / seed=$seed ==="
      if [[ -f "$model_out/model.pt" ]]; then
        echo "[run] model exists, skipping training"
      else
        "$PY" scripts/finetune/train.py \
          --train-jsonl "$DATA_DIR/train_${variant}.jsonl" \
          --backbone "$bb" \
          --out-dir "$model_out" \
          --epochs "$EPOCHS" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --seed "$seed" \
          --init-from "$pt_ckpt"
      fi
      if [[ -f "$eval_csv" ]] && [[ -f "${eval_csv%.csv}.summary.json" ]]; then
        echo "[run] eval exists, skipping"
      else
        "$PY" scripts/finetune/evaluate.py \
          --model-dir "$model_out" \
          --test-jsonl "$DATA_DIR/test.jsonl" \
          --out-csv "$eval_csv"
      fi
    done
  done
done

# 5. Summarize
echo ""
echo "=== [5] Summarizing (with vs without pre-train) ==="
"$PY" scripts/finetune/summarize.py \
  --results-dir "$EVAL_DIR" \
  --out-dir "$SUMMARY_DIR"

echo ""
echo "Done. Results in $OUT_DIR"
