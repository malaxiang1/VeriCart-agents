# Model Card: DAPO Shopping Agent Step-200

## Model

- Base: Qwen3.5-2B
- Adaptation: action-only LoRA SFT followed by online GRPO/DAPO-style updates
- Adapter: rank 8, alpha 16, all-linear target modules
- Vision tower: frozen for the primary text/tool experiment
- Export: FSDP shards -> standalone -> merged BF16 model

## Intended use

The model is an experimental shopping decision policy for a simulated product
catalog. It selects tool actions, verifies constraints, and terminates a
shopping workflow. It is not an autonomous payment system and must not be
connected to real purchasing without human confirmation, authentication,
inventory/price checks, and safety review.

## Evaluation

On the fixed 50-task ShopSimulator v2.1 split, strict purchase success is 66%
for DAPO step-200, compared with 58% for SFT and 60% for the short GRPO control.
Done and reward-valid rates are both 90% for DAPO; these lower reliability values
are part of the result.

## Limitations

The primary benchmark uses text/structured observations. It does not establish
visual grounding or multimodal accuracy. The validation split is small, the
same-budget GRPO-200 control is still needed, and the environment is simulated.
