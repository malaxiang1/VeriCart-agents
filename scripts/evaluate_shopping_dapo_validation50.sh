#!/usr/bin/env bash
set -euo pipefail

# Sequential, fixed-protocol Validation-50 comparison for SFT, GRPO, and DAPO.
# The historical HGPO evaluator is intentionally not part of this entry point.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/third_party/shopping-grpo-longhorizon"
PYTHON="$UPSTREAM/.venv/bin/python"
VLLM="$UPSTREAM/.venv/bin/vllm"
RUN_DIR="${1:-$ROOT/artifacts/evaluations/shopping_dapo_validation50}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${EVAL_PORT:-8010}"
DAPO_MODEL="${DAPO_MODEL:-$UPSTREAM/outputs/models/dapo-qwen35-25-merged}"
DAPO_LABEL="${DAPO_LABEL:-dapo_step25}"

mkdir -p "$RUN_DIR"

run_one() {
  local label="$1"
  local model="$2"
  local output="$RUN_DIR/$label"
  mkdir -p "$output"

  CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$model" \
    --served-model-name shopping-agent \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 24576 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    >"$output/vllm.log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' RETURN

  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$output/vllm.log" >&2
      return 1
    fi
    sleep 2
  done
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1

  "$PYTHON" "$UPSTREAM/scripts/evaluate_shop_benchmark.py" \
    --benchmark "$UPSTREAM/data/grpo/validation.jsonl" \
    --output "$output/trajectories.jsonl" \
    --summary "$output/summary.json" \
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
    >"$output/evaluate.log" 2>&1

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
}

run_one sft "$UPSTREAM/outputs/models/sft-merged-qwen35-2b-20k"
run_one grpo_step25 "$UPSTREAM/outputs/models/grpo-qwen35-matched25-standalone"
run_one "$DAPO_LABEL" "$DAPO_MODEL"

for pair in sft_vs_grpo sft_vs_dapo grpo_vs_dapo; do
  case "$pair" in
    sft_vs_grpo) base=sft; candidate=grpo_step25 ;;
    sft_vs_dapo) base=sft; candidate="$DAPO_LABEL" ;;
    grpo_vs_dapo) base=grpo_step25; candidate="$DAPO_LABEL" ;;
  esac
  "$PYTHON" "$ROOT/scripts/compare_shopping_benchmark.py" \
    --baseline "$RUN_DIR/$base/trajectories.jsonl" \
    --candidate "$RUN_DIR/$candidate/trajectories.jsonl" \
    --tasks "$UPSTREAM/data/grpo/validation.jsonl" \
    --output "$RUN_DIR/$pair.json"
done

echo "VALIDATION50_DAPO_COMPLETE=$RUN_DIR"
