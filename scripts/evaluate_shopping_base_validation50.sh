#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/third_party/shopping-grpo-longhorizon"
PYTHON="$UPSTREAM/.venv/bin/python"
VLLM="$UPSTREAM/.venv/bin/vllm"
RUN_DIR="${1:-$ROOT/artifacts/evaluations/shopping_base_validation50}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${EVAL_PORT:-8011}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to a local Qwen3.5-2B checkpoint}"

mkdir -p "$RUN_DIR/base"

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$BASE_MODEL" \
  --served-model-name shopping-agent \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 24576 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  >"$RUN_DIR/base/vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -100 "$RUN_DIR/base/vllm.log" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1

"$PYTHON" "$UPSTREAM/scripts/evaluate_shop_benchmark.py" \
  --benchmark "$UPSTREAM/data/grpo/validation.jsonl" \
  --output "$RUN_DIR/base/trajectories.jsonl" \
  --summary "$RUN_DIR/base/summary.json" \
  --base-url http://127.0.0.1:5700 \
  --model shopping-agent \
  --llm-base-url "http://127.0.0.1:$PORT/v1" \
  --api-key EMPTY \
  --max-steps 35 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 512 \
  --context-window 24576 \
  --context-compaction \
  >"$RUN_DIR/base/evaluate.log" 2>&1

echo "BASE_VALIDATION50_COMPLETE=$RUN_DIR"
