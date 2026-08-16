#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/third_party/shopping-grpo-longhorizon"
PYTHON="$UPSTREAM/.venv/bin/python"
MODEL="${MODEL:?set MODEL to a local Qwen3.5-2B checkpoint}"
ADAPTER="${ADAPTER:-$UPSTREAM/outputs/models/sft-lora-qwen35-2b-20k}"
MERGED="${MERGED:-$UPSTREAM/outputs/models/sft-merged-qwen35-2b-20k}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$UPSTREAM/src"

cd "$UPSTREAM"
"$PYTHON" scripts/train_lora_sft.py \
  --model "$MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER" \
  --max-length 20480 \
  --epochs 3 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --dtype bf16 \
  --gradient-checkpointing \
  --attention-implementation sdpa \
  --liger-kernel \
  --logging-steps 1 \
  --save-total-limit 3

exec "$PYTHON" scripts/merge_lora_adapter.py \
  --base-model "$MODEL" \
  --adapter "$ADAPTER" \
  --output "$MERGED" \
  --bf16
