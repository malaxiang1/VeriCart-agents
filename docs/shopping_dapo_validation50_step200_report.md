# Qwen3.5-2B Shopping DAPO Step-200 Report

## Protocol

- Benchmark: pinned ShopSimulator v2.1 `data/grpo/validation.jsonl`, 50 fixed task IDs.
- Reward: unchanged `shopsimulator-reward-v3`.
- Serving: the same vLLM 0.18.0 path, greedy decoding, 35 environment-turn cap,
  24,576-token evaluator context with compaction.
- Pairing: task IDs are matched before computing deltas; 10,000 paired bootstrap
  samples are used for the 95% intervals.
- Difference from the earlier DAPO report: the candidate is the resumed
  `global_step_200` checkpoint, not the 25-step checkpoint.

## Results

| Policy | Strict success | Purchase success | Mean reward-v3 | Done rate | Mean turns | Reward-valid rate |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B base | 0/50 (0%) | 0/50 (0%) | -0.15513 | 28% | 5.44 | 26% |
| SFT | 29/50 (58%) | 29/50 (58%) | 0.39085 | 96% | 11.68 | 96% |
| GRPO, step 25 | 30/50 (60%) | 30/50 (60%) | 0.43846 | 96% | 11.14 | 94% |
| DAPO, step 200 | 33/50 (66%) | 33/50 (66%) | 0.51838 | 90% | 10.14 | 90% |

The raw base model has no successful purchase and 34 invalid-action-limit
episodes. Action-only SFT contributes the largest capability jump: `0% -> 58%`
strict success, a paired +58 point delta with 95% interval `[+44,+72]`. DAPO
step-200 reaches `0% -> 66%` versus the raw base, a paired +66 point delta with
interval `[+52,+78]`; because the baseline rate is zero, no relative percentage
is reported for these two comparisons.

Paired SFT -> DAPO strict-success delta is `+8` percentage points (`+13.79%`
relative), with a 95% bootstrap interval of `[+2,+16]` points. Paired GRPO ->
DAPO is `+6` points (`+10.00%` relative), with a 95% interval of `[0,+14]`
points. Mean reward-v3 increases by `+0.12752` (`+32.63%`) over SFT, with
paired interval `[+0.02127,+0.25304]`; over GRPO it increases by `+0.07992`
(`+18.23%`), with interval `[-0.00142,+0.18232]`.

The candidate has five invalid-action-limit episodes and five unknown reward
records, compared with two invalid-action-limit episodes for each baseline. The
strict-success denominator remains the full 50 tasks, so the improvement is
usable as a benchmark result, but the lower done/reward-valid rate is a material
reliability caveat. The result supports an accuracy improvement over SFT and a
positive but borderline improvement over matched GRPO; it does not justify a
claim of uniformly better reliability.

## Training evidence

The run resumed from the complete step-25 DAPO checkpoint and finished at
`global_step_200` on two RTX 3090 GPUs. It used partial-valid dynamic sampling,
up to three generation batches, token-mean loss, and DAPO clip-higher `0.20/0.28`.
The final training log records 45 skipped sampling updates, finite actor
gradients, and final in-training validation reward `0.475875`.

## Artifacts

- `artifacts/evaluations/shopping_dapo_validation50_step200/{sft,grpo_step25,dapo_step200}/summary.json`
- `artifacts/evaluations/shopping_dapo_validation50_step200/{sft_vs_dapo,grpo_vs_dapo}.json`
- `artifacts/evaluations/shopping_base_validation50/base/summary.json`
- `artifacts/evaluations/shopping_base_validation50/{base_vs_sft,base_vs_grpo,base_vs_dapo}.json`
- `third_party/shopping-grpo-longhorizon/outputs/models/dapo-qwen35-25/global_step_200`
- `third_party/shopping-grpo-longhorizon/outputs/models/dapo-qwen35-200-standalone`
- `third_party/shopping-grpo-longhorizon/outputs/models/dapo-qwen35-200-merged`

## Resume-Safe Claim

> Built a Qwen3.5-2B long-horizon shopping Agent post-training pipeline on 2x
> RTX 3090 with DAPO-style partial-valid dynamic sampling, then trained to 200
> optimizer steps and evaluated on 50 fixed ShopSimulator tasks. DAPO step-200
> improved strict purchase success from 0% raw base / 58% SFT / 60% matched GRPO
> to 66% and
> increased mean reward-v3 by 32.6% over SFT (paired 95% CI +2 to +16 points
> for strict success).
