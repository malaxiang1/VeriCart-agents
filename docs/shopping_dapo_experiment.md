# Shopping DAPO Experiment Contract

This is the active online-RL experiment for the Qwen3.5-2B shopping Agent. The
historical HGPO branch is archived and is not used for method selection.

## Question

Does DAPO-style dynamic sampling improve the reliability of GRPO updates in the
ShopSimulator v2.1 tool environment, where an individual rollout can be invalid
or where all four rollouts in a group can receive the same terminal reward?

## Matched setup

- Base policy: the existing merged Qwen3.5-2B action-only SFT checkpoint.
- Hardware: exactly two RTX 3090 GPUs, the pinned `torch`/`veRL`/`vLLM` runtime.
- Data: the launcher-locked GRPO train and Validation-50 splits; Final-200 is
  untouched during training and checkpoint selection.
- Rollout budget: four trajectories per prompt, four tool workers, max 35
  environment turns, 8K runtime context, and 25 optimizer updates for the
  bounded main run.
- Reward: unchanged ShopSimulator reward-v3 and the upstream evaluation code.

## DAPO change

The launcher enables bounded dynamic sampling with `selection_mode=partial_valid`.
At most three generation batches are attempted per update. All-equal groups and
sampling-invalid trajectories are filtered; valid trajectories are packed into an
update when at least two remain. The actor uses token-mean loss and asymmetric
clip-higher (`0.20` lower, `0.28` upper). No HGPO credit, reward shaping, or
Final-200 feedback is enabled.

## Acceptance metrics

Training diagnostics must show finite gradients and optimizer updates, together
with accepted/generated groups, resampling batches, sampling-invalid rate, and
rollouts per accepted group. The benchmark report uses the unchanged primary
metrics: strict `gold_purchase` success, valid purchase success, mean reward-v3,
done rate, average turns, repeat-action rate, and paired task-level bootstrap
confidence intervals against the SFT/GRPO baseline.

The result is an improvement only if the same fixed-denominator evaluation shows
positive strict-success and purchase-success deltas without a material legality
or completion regression. A Validation-50 gain alone is not sufficient.
