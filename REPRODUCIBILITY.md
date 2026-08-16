# Reproducibility Notes

## Fixed contract

- Upstream commit: `c3c178595eea835c18ba4515d553025014e52656`
- Environment: ShopSimulator v2.1 / `shopsimulator-reward-v3`
- RL train rows: 1000
- Validation rows: 50 fixed task IDs
- GPUs: 2x RTX 3090, 24 GiB each
- Rollout: 4 trajectories per prompt, temperature 0.7, top-p 0.9
- Evaluation: temperature 0, top-p 1, max 35 environment turns
- DAPO: partial-valid dynamic sampling, at most 3 generation batches,
  clip range 0.20/0.28

## Hashes and manifests

The full local run contains `run_manifest.json` files with model/config/data
hashes. The public package includes only lightweight result summaries because
raw datasets, model weights, and trajectories are not redistributed.

## Expected evidence

A valid reproduction should produce:

1. finite actor loss and gradient metrics;
2. FSDP checkpoints at the configured save frequency;
3. a standalone or merged model that vLLM can load;
4. one summary JSON per policy with all 50 expected task IDs;
5. paired comparison JSON with task-level transitions and bootstrap intervals.

## Resource warning

The 200-step run took multiple hours on two RTX 3090 GPUs. Start with
`--steps 2` or `--dry-run` to validate the environment before spending compute.
