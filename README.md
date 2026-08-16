# VeriCart-Agent

Qwen3.5-2B long-horizon shopping Agentic RL on two RTX 3090 GPUs.

This repository packages the reproducible project code, configuration, evaluation
protocol, lightweight metrics, and interview documentation. Model weights,
datasets, trajectories, virtual environments, and runtime downloads are
intentionally excluded.

## What It Solves

The target is a shopping policy that can turn a natural-language request into a
verified multi-step tool workflow:

```text
understand constraints -> search -> inspect products -> select options
-> verify price/attributes -> purchase or stop safely
```

The project studies how to make Agentic RL updates robust when long-horizon tool
rollouts contain invalid actions, unverifiable terminal rewards, or all-equal
groups with no relative learning signal.

## Result

The same fixed 50-task ShopSimulator v2.1 validation split was used for every
policy. The reward contract is the pinned `shopsimulator-reward-v3` evaluator.

| Policy | Strict purchase success | Mean reward-v3 | Done rate | Reward-valid rate |
|---|---:|---:|---:|---:|
| Raw Qwen3.5-2B | 0% | -0.15513 | 28% | 26% |
| Action-only SFT | 58% | 0.39085 | 96% | 96% |
| Native GRPO, step 25 | 60% | 0.43846 | 96% | 94% |
| DAPO-style, step 200 | **66%** | **0.51838** | 90% | 90% |

DAPO step-200 improves strict success by `+8` percentage points over SFT
(paired 95% bootstrap CI `[+2,+16]`) and by `+6` points over the short GRPO
baseline (CI `[0,+14]`). Mean reward-v3 improves by 32.63% over SFT.
The lower done/reward-valid rate is a required caveat; this is not a claim of
uniformly better reliability.

## Method

- Qwen3.5-2B multimodal-capable base, with the vision tower frozen in this text/tool benchmark.
- Action-only LoRA SFT warm start.
- Four multi-turn tool rollouts per prompt.
- Native GRPO group-relative advantage without a critic.
- DAPO-style `partial_valid` dynamic sampling: filter invalid/all-equal groups,
  bounded resampling, and skip updates without valid relative signal.
- DAPO clip-higher range: `clip_ratio_low=0.20`, `clip_ratio_high=0.28`.
- Token-mean actor loss, LoRA rank 8, FSDP sharding, gradient checkpointing,
  CPU parameter/optimizer offload, and vLLM rollouts.

This is a DAPO-style engineering adaptation, not a claim to reproduce every
detail of the DAPO paper. The primary benchmark is text/structured shopping
interaction; visual grounding improvement is not claimed.

## Repository Layout

```text
configs/       2x3090 RL and AgentLoop configuration
scripts/       training, evaluation, export, and comparison entry points
groundedvision/paired benchmark comparison and bootstrap utilities
tests/         focused evaluation and launcher tests
patches/       pinned upstream shopping and veRL runtime patches
results/       summary JSON and paired comparison JSON, no raw trajectories
docs/          experiment report, claim audit, and interview guide
```

## Reproduce

### 1. Prepare the pinned upstream environment

The runtime depends on the public upstream project at commit
`c3c178595eea835c18ba4515d553025014e52656`:

```bash
git clone https://github.com/YYHDBL/shopping-grpo-longhorizon.git \
  third_party/shopping-grpo-longhorizon
cd third_party/shopping-grpo-longhorizon
git checkout c3c178595eea835c18ba4515d553025014e52656
```

Install the upstream environment according to its own setup instructions. The
tested stack is veRL 0.8.0, vLLM 0.18.0, PyTorch 2.10.0+cu128, Transformers
5.15.0.dev0, and FlashInfer 0.6.6.

### 2. Apply the project patches

Apply the upstream source patch from the upstream repository root:

```bash
git apply ../../patches/shopping-grpo-c3c1785.patch
```

Apply the veRL runtime patch from the environment's site-packages root. The
patch was generated against the original veRL 0.8.0 `ray_trainer.py`:

```bash
cd .venv/lib/python3.12/site-packages
patch -p1 < ../../../../../../patches/verl-0.8.0-vericart.patch
```

The exact relative path can differ with Python minor version; use
`python -c "import verl; print(verl.__file__)"` to locate the package first.

### 3. Run preflight or training

The Qwen3.5-2B weights are intentionally not bundled. Set `MODEL` to a local
checkpoint for SFT; pass the merged SFT checkpoint to the RL launcher. A 2x3090
run requires the ShopSimulator service on `http://127.0.0.1:5700`.

```bash
MODEL=/path/to/Qwen3.5-2B bash scripts/run_shopping_sft_2x3090.sh
python scripts/run_shopping_rl_2x3090.py \
  --method dapo \
  --model third_party/shopping-grpo-longhorizon/outputs/models/sft-merged-qwen35-2b-20k \
  --dry-run
python scripts/run_shopping_rl_2x3090.py \
  --method dapo \
  --steps 200 \
  --model third_party/shopping-grpo-longhorizon/outputs/models/sft-merged-qwen35-2b-20k \
  --output third_party/shopping-grpo-longhorizon/outputs/models/dapo-qwen35-200
```

### 4. Run the fixed evaluation

```bash
BASE_MODEL=/path/to/Qwen3.5-2B \
  bash scripts/evaluate_shopping_base_validation50.sh \
  results/validation50/base

DAPO_MODEL=/path/to/dapo-qwen35-200-merged \
DAPO_LABEL=dapo_step200 \
bash scripts/evaluate_shopping_dapo_validation50.sh \
  results/validation50_step200
```

The evaluation uses greedy decoding, 35 environment turns, a 24576-token
evaluator context with compaction, and the same 50 task IDs for every model.

## Evidence Files

- [Step-200 report](docs/shopping_dapo_validation50_step200_report.md)
- [Claim audit](docs/shopping_dapo_step200_claim_validation.md)
- [Interview guide](docs/interview_project_guide.md)
- [Lightweight validation results](results/validation50/)

## Scope and Limitations

- ShopSimulator v2.1 is an open-source benchmark protocol, not the WebShop official leaderboard.
- The current main benchmark is text/structured tool interaction; it does not establish visual grounding gains.
- Validation-50 is suitable for a reproducible engineering comparison, not a broad SOTA claim.
- GRPO step-200 is the most important missing same-budget control for isolating DAPO from training duration.
- DAPO improves success but currently lowers done/reward-valid rate; future work must address protocol robustness.

## Attribution

See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) before publishing. The
upstream repository is kept as a pinned dependency and is not represented as
original project code.
