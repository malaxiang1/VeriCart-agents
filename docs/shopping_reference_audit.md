# Shopping Reference Audit

Audit date: 2026-08-07

## Pinned Source

- Repository: `YYHDBL/shopping-grpo-longhorizon`
- Commit: `c3c178595eea835c18ba4515d553025014e52656`
- Environment manifest: ShopSimulator Environment v2.1, Reward v3
- Source license: no repository `LICENSE` file was present at the pinned commit

The checkout under `third_party/` remains unmodified. VeriCart-Agent owns its
PAD implementation, runtime port, launchers and tests. The reference checkout is
not presented as redistributable project code while its license is unspecified.

## Data Verification

| Asset | Rows | SHA-256 |
|---|---:|---|
| SFT train | 379 | `8cd1f72130b3c781d5ffe08fe3e399b2a9e45d204e3f3bd0d8e677d1b51c8ec5` |
| SFT validation | 49 | `f8ae506d0fa9d1526342a9f717da24922c8a55776d076a296698abac4fde05b3` |
| GRPO train parquet | 1,000 | `e4b4765b67efcc064ba4e656db625a812a34cbcff00da0e23d6a3df8aac5fdd4` |
| GRPO validation parquet | 50 | `9aa370f00d7ead942e47cf9aed4ab0c55cd7426220f4603f4fed5dc949ea2788` |
| Final-200 IDs | 200 | `2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208` |

The SFT train/validation, GRPO train/validation and Final-200 task-ID sets are
pairwise disjoint. The main RL launcher recomputes the training hashes and the
RL/Final overlap before every run.

The reference README contains one inconsistent hardware-table statement saying
"448 training rows". The actual files, metadata and SFT run manifest consistently
specify 379 train plus 49 validation rows. VeriCart-Agent uses the files and
hashes, not the inconsistent prose.

## Runtime Verification

- Product archive expanded to 23,421 products and matched manifest SHA-256
  `57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f`.
- Search index built successfully.
- A real API reset/search/release smoke completed on RL-train task 19089.
- The Qwen3.5-2B vLLM port loaded 4.25 GiB of weights on an RTX 3090 and generated
  `OK` through the Qwen3.5 multimodal architecture.
- The upstream CUDA 13 runtime was incompatible with driver 570.211. The verified
  port is torch 2.10.0+cu128, vLLM 0.18.0, veRL 0.8.0, the pinned Transformers
  commit, and matching FlashInfer 0.6.6 packages.

## Test Trust Ranking

Project-owned PAD and fixed-denominator evaluation tests: 13/13 passed.

Reference critical-path subset: 47/47 passed, covering:

- action validation;
- context compaction;
- ShopAgentEnv client behavior;
- shopping tools;
- reward-group dynamic sampling and its veRL patch;
- deterministic benchmark aggregation.

The reference full suite is not green at the pinned commit. It has one collection
error (`validate_reward_components` is imported by a test but has never existed in
the tracked runtime history) plus 13 failures concentrated in older observation,
rollout fake-environment and terminal-state contracts. The same issue exists in
the parent commit, so it was not introduced by the latest ablation commit or by
the 3090 port.

Consequences:

- Do not cite "all upstream tests pass".
- Require a real end-to-end AgentLoop smoke before a long RL launch.
- Treat published reference metrics as external results until locally reproduced.
- Keep Final-200 untouched until matched validation selects one checkpoint per
  method.

## Metric Status

The reference reports Base 0.0%, SFT 60.5%, and GRPO step-100 62.0% strict success
on its Final-200 split. These are reference-reported numbers, not results produced
by VeriCart-Agent. The project will report its own numbers only after complete
matched evaluation artifacts exist.
