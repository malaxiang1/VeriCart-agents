# DAPO Step-200 Claim Validation

| Claim | Metric | Observation | Verdict |
|---|---|---|---|
| The 200-step pipeline is reproducible | checkpoint and export integrity | `global_step_200` contains both FSDP actor shards; standalone and merged BF16 exports succeeded | supported |
| SFT is necessary for tool-protocol acquisition | Validation-50 strict success | raw base 0% -> SFT 58%, paired 95% CI [+44,+72] | supported |
| DAPO step-200 improves over SFT | Validation-50 strict success | 58% -> 66%, +8 points, paired 95% CI [+2,+16] | supported |
| DAPO step-200 improves over matched GRPO | Validation-50 strict success | 60% -> 66%, +6 points, paired 95% CI [0,+14] | supported, borderline |
| DAPO step-200 improves terminal reward over SFT | mean reward-v3 | 0.39085 -> 0.51838, +32.63%, paired CI [+0.02127,+0.25304] | supported |
| DAPO step-200 improves terminal reward over GRPO | mean reward-v3 | 0.43846 -> 0.51838, +18.23%, paired CI [-0.00142,+0.18232] | inconclusive |
| DAPO improves reliability on every axis | done/reward-valid rate | candidate 90%/90%, baselines 96%/96% (SFT) and 96%/94% (GRPO) | refuted |

The strict-success comparison uses the complete 50-task denominator. Five DAPO
episodes ended at the evaluator's invalid-action limit, so the accuracy claim
must be reported together with the lower completion and reward-valid rates.
The raw base model has 34 invalid-action-limit episodes; its 0% success is a
tool-protocol capability floor, not evidence that the pretrained model lacks
general product knowledge.
