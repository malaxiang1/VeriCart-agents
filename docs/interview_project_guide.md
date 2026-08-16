# Qwen3.5-2B Shopping Agentic RL 项目：简历与面试完整指南

> 本文是项目的面试准备文档，不是论文。所有已经完成的数字都来自仓库中的真实日志和评测产物；没有测过的能力会明确标记为“未验证”，不要在面试中把规划说成结果。

## 1. 项目一句话

在两张 RTX 3090 上，基于 Qwen3.5-2B 多模态底座搭建一个长流程购物 Agent 的后训练闭环：用 Action-only SFT 让模型学会工具协议，再用原生 GRPO 和 DAPO-style dynamic sampling 做在线 Agentic RL，最后在固定的 ShopSimulator v2.1 Validation-50 上进行任务级、配对统计评测。

项目的核心结果是：原始 Base 模型 strict purchase success 为 0%，SFT 为 58%，匹配的 GRPO step-25 为 60%，DAPO step-200 为 66%。DAPO 相对 SFT 提升 8 个百分点，mean reward-v3 从 0.39085 提升到 0.51838（+32.63%）。

必须同时披露：DAPO 的 done rate/reward-valid rate 为 90%，低于 SFT 的 96%，所以结论是“任务成功率提升，但可靠性并非所有维度都提升”。

## 2. 简历版本

### 2.1 中文一行版

**Qwen3.5-2B 长流程购物 Agentic RL**：在 2x RTX 3090 上实现 Action-only SFT、在线多轮工具交互、原生 GRPO 与 DAPO-style partial-valid dynamic sampling；在固定 50-task ShopSimulator v2.1 评测中，将 strict purchase success 从 SFT 的 58% 提升至 DAPO step-200 的 66%，并完成 FSDP/LoRA 导出、BF16 合并和 paired bootstrap 评测。

### 2.2 英文简历版

**Long-Horizon Shopping Agentic RL with Qwen3.5-2B**: Built an end-to-end post-training pipeline on 2x RTX 3090 GPUs, including action-only SFT, multi-turn tool rollouts, native GRPO, DAPO-style partial-valid dynamic sampling, FSDP/LoRA export, and deterministic ShopSimulator evaluation. Improved strict purchase success from 58% (SFT) to 66% (DAPO, 200 updates) on a fixed 50-task split, with paired task-level bootstrap confidence intervals.

### 2.3 更保守、更适合大厂简历的版本

**Agentic RL Shopping System**：复现并改造开源 ShopSimulator Agent 训练闭环，针对长流程工具轨迹中的 invalid/all-equal reward group，实现 partial-valid dynamic sampling 和 DAPO clip-higher；在固定 Validation-50 上完成 Base/SFT/GRPO/DAPO 四阶段对照，DAPO step-200 达到 66% strict success，较 SFT +8pp。

### 2.4 不要这样写

- 不要写“在官方 WebShop 榜单 SOTA”。本项目使用的是上游开源 ShopSimulator v2.1 split，不是 WebShop 官方 leaderboard。
- 不要写“多模态视觉能力提升”。本次主 benchmark 的 observation 是文本工具状态，Qwen3.5 vision tower 冻结，没有视觉 ablation。
- 不要写“DAPO 在所有指标上提升”。done rate 和 reward-valid rate 反而下降。
- 不要把训练中间的 reward 直接当成最终购买成功率。
- 不要把 50 个任务的结果说成大规模统计结论；应称为 fixed Validation-50 evidence。

## 3. 事实卡片：面试前必须记住

| 项目 | 事实 |
|---|---|
| 基础模型 | Qwen3.5-2B，权重路径由环境变量 `MODEL` 或启动参数提供 |
| SFT 起点 | `sft-merged-qwen35-2b-20k`，Action-only LoRA SFT 后合并模型 |
| RL 框架 | veRL 0.8.0 + Ray + vLLM 0.18.0 |
| 设备 | 2x NVIDIA RTX 3090，单卡 24 GiB |
| 训练方式 | LoRA rank 8，alpha 16，target modules `all-linear`，vision tower frozen |
| RL optimizer | AdamW 路径，学习率 `1e-6`，gradient checkpointing，参数/optimizer offload |
| rollout | 每个 prompt 4 条 trajectory，temperature 0.7，top-p 0.9，多轮工具调用 |
| RL train batch | prompt batch size 2，DAPO actor mini-batch size 1 |
| response cap | 训练 response max 6144，rollout engine max model length 8192 |
| evaluator context | 24576 token window，开启 context compaction；这是外部评测协议，不等同于训练 rollout cap |
| 训练集 | RL train parquet/jsonl 对应 1000 条 prompt |
| 验证集 | 固定 50 条 Validation-50 task ID |
| 最终补充集 | Final-200，训练未触碰；本次主结论不依赖它 |
| reward | `shopsimulator-reward-v3` |
| active algorithm | DAPO-style dynamic sampling；HGPO 已退出主线 |
| DAPO clip | low `0.20`，high `0.28` |
| DAPO updates | 从 step-25 checkpoint 恢复，完成到 global step 200 |
| 最终 in-training validation | mean reward `0.475875` |
| 外部 Validation-50 | Base 0%，SFT 58%，GRPO-25 60%，DAPO-200 66% |
| 统计 | fixed denominator 50，task-ID paired bootstrap，10000 samples，seed 42 |

## 4. 四阶段结果怎么讲

| Policy | Strict success | Purchase success | Mean reward-v3 | Done rate | Reward-valid rate | 解释 |
|---|---:|---:|---:|---:|---:|---|
| Raw Base | 0/50 | 0/50 | -0.15513 | 28% | 26% | 没学会严格工具协议，不能作为“知识能力为零”的证明 |
| Action-only SFT | 29/50 | 29/50 | 0.39085 | 96% | 96% | 最大增益来自工具调用格式、状态跟踪和购买流程 |
| Native GRPO step-25 | 30/50 | 30/50 | 0.43846 | 96% | 94% | 短预算下相对 SFT 只 +2pp |
| DAPO step-200 | 33/50 | 33/50 | 0.51838 | 90% | 90% | 相对 SFT +8pp，相对 GRPO +6pp，但可靠性有代价 |

### 4.1 可以公开的数字

- Base -> SFT strict success: `0% -> 58%`，paired 95% CI `[+44,+72]` pp。
- SFT -> DAPO strict success: `58% -> 66%`，+8pp，paired 95% CI `[+2,+16]` pp。
- GRPO -> DAPO strict success: `60% -> 66%`，+6pp，paired 95% CI `[0,+14]` pp，属于正向但 borderline。
- SFT -> DAPO mean reward: `0.39085 -> 0.51838`，绝对 +0.12752，相对 +32.63%，paired CI `[+0.02127,+0.25304]`。

### 4.2 为什么不把 Base 的 reward 写成“提升 434%”

Base mean reward 是负数，且相对变化的分母接近“失败惩罚”，百分比没有稳定解释。面试中应写绝对 delta 和成功率，不写夸张倍数。

### 4.3 为什么 DAPO reward 提升但 done rate 降低

这是两个不同维度。reward 只看终局任务结果及 reward-v3；done rate 还反映轨迹是否正常闭环结束。DAPO 可能更愿意探索不同搜索路径，成功轨迹变多，但仍有一些轨迹在 invalid action guard 或循环后耗尽预算。因此项目结论不是“全面更可靠”，而是“准确任务成功率提高，同时需要继续修复协议可靠性”。

## 5. 系统架构

```text
Qwen3.5-2B base
        |
        | action-only LoRA SFT
        v
merged SFT policy ----------------------+
        |                                |
        | native GRPO baseline           | DAPO-style RL
        v                                v
  4 rollouts/prompt                 4 rollouts/prompt
        |                                |
        +-------- ShopSimulator v2.1 ---+
                         |
                 reward-v3 / terminal utility
                         |
        +----------------+----------------+
        |                                 |
  native group advantage          partial-valid sampler
                                      + clip-higher
                         |
                    LoRA actor update
                         |
                 FSDP checkpoint every 25 steps
                         |
           export -> standalone -> merged BF16
                         |
             fixed Validation-50 + paired bootstrap
```

### 5.1 一条 trajectory 的完整流程

1. 从 RL train parquet 取出一个用户购物需求和任务 ID。
2. Agent 收到初始 observation，必须使用规定的 search/open/select/buy 工具。
3. 每次工具调用都由 ShopSimulator 返回新的 structured/text observation。
4. Agent 继续做搜索、翻页、打开商品、查看属性、选择规格，直到购买或结束。
5. 环境在终局计算 Reward-v3：商品是否匹配、属性是否满足、价格是否满足、是否完成购买、是否循环/超步数。
6. veRL 收集 4 条 trajectory，计算 group-relative advantage。
7. DAPO sampler 过滤 invalid/all-equal group，必要时最多追加 3 个 generation batch。
8. 有效 group 被打包进入 actor loss；无效更新被跳过并记录统计。
9. 每 25 步保存 FSDP model/optimizer/extra state，并执行 validation。
10. 最终 actor 被合并为 standalone BF16 模型，使用确定性 greedy evaluator 重跑 50 个固定任务。

## 6. GRPO 原理：面试必须会推导

### 6.1 GRPO 解决什么问题

GRPO（Group Relative Policy Optimization）是 PPO 的一种 critic-free 变体。对同一个 prompt 采样一组 response，而不是训练一个额外的 value model。用同组 response 的 reward 均值作为 baseline，估计相对优势，从而降低 prompt 难度差异带来的方差，并节省 critic 的显存和训练成本。

在本项目中，一个购物 prompt 生成 `G=4` 条多轮工具轨迹：

```text
r_i = terminal reward of trajectory i
mu = mean(r_1, ..., r_G)
A_i = r_i - mu                 # 本项目关闭 std normalization
```

配置中 `algorithm.adv_estimator=grpo`，`norm_adv_by_std_in_grpo=false`。不要在面试中说本项目使用了“均值加标准差归一化”，那是很多 GRPO 实现的默认选项，但不是当前配置。

### 6.2 PPO/GRPO 的 ratio 和 clipping

对 response 中每个 token，策略比值是：

```text
rho_t(theta) = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
```

经典 PPO clipped objective：

```text
L_clip = E[min(rho_t * A_t,
               clip(rho_t, 1-epsilon, 1+epsilon) * A_t)]
```

GRPO 仍然使用 PPO-style ratio 和 clipping，只是 `A_t` 由同一 prompt 的 group reward 产生，而非 GAE critic。

### 6.3 本项目的 token-mean loss

多轮 Agent response 很长，不同 trajectory 的 token 数差异很大。如果直接按 sequence 求和，长轨迹会获得更大梯度权重。本项目使用 `loss_agg_mode=token-mean`，让有效 response token 的贡献先按 token 平均，再聚合 batch，从而减少长度偏置。

面试回答要补充：token-mean 不等于“完全解决长轨迹偏置”。它只是改变 loss aggregation；trajectory 本身仍然可能因为行动次数、上下文长度和 reward 结构产生优化偏置。

### 6.4 为什么不用 critic

- 两张 24 GiB 3090 的显存预算紧，2B 多模态模型的 actor/rollout 已经占用大量显存。
- GRPO 可以使用同 prompt 的多条 response 做相对基线，省掉 value model 参数、optimizer state 和 value loss。
- 购物 reward 是稀疏的终局 reward，critic 在短预算阶段容易不稳定，先用 group-relative signal 更简单。

代价是：group reward 质量直接决定 advantage 质量；当 4 条轨迹全部成功、全部失败或 reward 无效时，更新信号可能退化，这正是 DAPO sampler 要解决的问题。

## 7. DAPO-style 改进：本项目真正做了什么

### 7.1 Dynamic sampling

普通 whole-group GRPO 要求一个 group 的 4 条轨迹一起进入 update。长流程工具环境中，经常出现：

- 一个轨迹被工具 guard 拦截；
- 一个轨迹 reward unverifiable；
- 四条轨迹 reward 完全相同，没有相对学习信号；
- 一条轨迹是基础设施失败，但其他轨迹仍有合法终局。

本项目启用：

```text
shopping_dynamic_sampling.enable=true
selection_mode=partial_valid
rollout_n=4
max_num_gen_batches=3
max_consecutive_skipped_updates=10
```

逻辑是：

1. 先生成一个 prompt 的 4 条 trajectory。
2. 检查 reward version、reward_valid、terminal utility、purchase_success 和 all-equal 状态。
3. 如果 group 中有至少两个可用成员，则保留有效 peer 进行 update。
4. 如果有效成员不足，则最多继续生成 3 个 batch 尝试恢复。
5. 仍然无法形成有效相对信号时，跳过 optimizer update，并记录 `skipped_updates_total`。

这里的“跳过”不是吞掉错误，而是防止无意义或污染的 advantage 更新模型。

### 7.2 DAPO clip-higher

本项目使用：

```text
clip_ratio_low = 0.20
clip_ratio_high = 0.28
```

相比对称的 `[0.20, 0.20]`，正向更新侧允许更大的 ratio 上界，目标是让高优势 response 有更强的学习空间，同时保留负向侧的保守约束。

面试时要说清楚：这不是“把 reward 直接放大”，而是改变 PPO surrogate objective 对 policy ratio 的截断区间；它仍然受到 old policy ratio 和 advantage 的共同约束。

### 7.3 本项目 DAPO 的边界

不要说“完整复现了论文全部 DAPO”。本项目吸收的是 DAPO-style 的两类工程机制：dynamic sampling 和 clip-higher，并适配了 ShopSimulator 的 invalid trajectory。没有宣称实现所有 DAPO 论文细节，也没有用 DAPO 论文 benchmark 结果替代本项目测评。

## 8. Reward-v3 设计

### 8.1 Reward 类型

| 类型 | 含义 | 默认 reward |
|---|---|---:|
| `gold_purchase` | 精确目标商品、约束满足且正确购买 | 1.0 |
| `valid_alternative_purchase` | 目标商品不同，但满足硬约束和偏好阈值 | 0.55 |
| `partial_alternative_purchase` | 部分匹配，但未完全满足偏好 | `-0.30 + 0.55 * match_score`，上限 0.25 |
| `wrong_purchase` | 购买违反硬约束 | -0.85 |
| `repeat_loop` | 重复动作/循环 | -0.65 |
| `max_steps` | 到达环境步数上限 | -0.50 |
| `graceful_stop` | 探索后确认没有合适商品并正常停止 | -0.15 |
| `early_abstain` | 没有完成最低探索就放弃 | -0.35 |
| `reward_unverifiable` | 关键 reward 证据无法验证 | 0.0 |

### 8.2 为什么 strict success 只认 gold_purchase

如果把 partial purchase 或 valid alternative 也算完全成功，会把“买到了一个有点像的商品”和“准确满足用户需求”混为一谈。strict success 采用最严格的可解释口径：环境正常结束、reward-v3 有效、终止类型是 gold_purchase、purchase_success 为 true。

同时保留 purchase success 和 mean reward，用来分析模型是否有“合法替代”能力以及软匹配质量。

### 8.3 Reward hacking 风险

可能的 reward hacking 包括：

- 只搜索容易得到高 match score 的商品，不验证关键属性；
- 反复点击或翻页，等待偶然的商品匹配；
- 选择合法替代商品而逃避精确目标商品；
- 利用 reward unverifiable 的空值或解析漏洞。

本项目用 hard gates、reward_valid、termination reason、action guard 和 fixed benchmark 约束这些行为，但仍然观察到 DAPO 的 invalid-action-limit 增多，所以 reward hacking/协议鲁棒性仍是后续方向。

## 9. 多模态到底在哪里

### 已经具备的部分

- 底座 Qwen3.5-2B 是 multimodal-capable model。
- 模型配置保留 vision tower 和 multimodal processor。
- 训练使用 LoRA，`freeze_vision_tower=true`，因此视觉骨干没有被破坏。
- 商品环境保留了 image/URL 相关字段和后续视觉工具扩展接口。

### 本次没有验证的部分

- ShopSimulator 主 benchmark 的 observation 是文本/结构化工具状态。
- 本次没有让模型读取商品图片完成视觉属性判断。
- 没有视觉 ablation，也没有在视觉 benchmark 上报告提升。

面试正确说法：

> 这是一个多模态底座上的 Agentic RL shopping pipeline。本次主实验测量的是文本工具交互和长流程决策能力，视觉 tower 冻结；视觉 grounding 是明确的下一阶段扩展，不把当前结果冒充视觉提升。

## 10. 评测和统计

### 10.1 为什么固定分母是 50

每个 policy 都在同一组 50 task ID 上运行；即使某条 trajectory invalid 或 evaluator 没有正常结束，也保留为该 task 的失败，而不是从分母删除。这样避免“只统计完成任务”导致成功率虚高。

### 10.2 为什么做 paired comparison

不同 task 难度差异很大。把 SFT 和 DAPO 的成功率简单相减没有体现“哪几个任务发生改变”。配对比较按 task ID 对齐：

```text
task 3413: SFT fail -> DAPO success
task 3904: SFT success -> DAPO success
task 12345: SFT fail -> DAPO fail
```

然后对每个 task 的 candidate-baseline 差值做 bootstrap，保持 task pairing。

### 10.3 95% CI 是什么

当前实现使用 10,000 次、有放回的 task-level resampling，seed 42，取 2.5% 和 97.5% percentile。它不是“模型 95% 概率在这个区间”，而是有限样本下 delta 估计的不确定性区间。

### 10.4 50 个任务够不够

够做工程 checkpoint comparison 和面试展示，但不够支撑大规模 SOTA 结论。下一步应该：

- 增加 untouched held-out split；
- 多 seed 或重复 rollout；
- 分析 task difficulty 和 failure type；
- 报告 Wilson/binomial interval 或 task bootstrap；
- 公开完整 trajectory 和 config。

## 11. 训练工程

### 11.1 为什么 2x3090 能跑

- Qwen3.5-2B 参数量相对可控。
- LoRA 只训练低秩 adapter，冻结 vision tower。
- gradient checkpointing 降低 activation memory。
- FSDP sharding 将参数/梯度分片到两张卡。
- optimizer/parameter offload 把部分状态放到 CPU。
- `use_remove_padding=true` 减少 padding token 浪费。
- vLLM rollout 和 actor update 采用低 GPU memory utilization，避免 rollout 服务挤爆 actor。

### 11.2 为什么 batch size 只有 2

每条 shopping trajectory 是多轮工具调用，token 长度和 observation 长度变化很大。显存瓶颈不是 prompt 数量，而是 response token、KV cache 和 FSDP actor update 的峰值。batch size 2 是在 24 GiB 卡上实测稳定的折中。

### 11.3 为什么训练那么慢

每个 optimizer step 不等于一次普通文本 forward：

```text
4 trajectories/prompt
  * multi-turn tool calls
  * environment HTTP round trips
  * vLLM generation
  * log-prob recomputation
  * FSDP actor update
  * occasional resampling
  * checkpoint/validation every 25 steps
```

因此 step-200 是 200 个有效 global steps，不等同于 200 条样本。

### 11.4 如何保证可恢复

每 25 step 保存：

- FSDP actor model shards；
- optimizer shards；
- RNG state；
- learning-rate scheduler state；
- tokenizer/processor/config。

veRL 的 `resume_mode=auto` 会扫描 `global_step_*`，本项目实际从 step-25 恢复到 step-100，再恢复完成到 step-200。中间 ShopSimulator 服务发生 slot exhaustion 时重启环境并从完整 checkpoint 恢复，没有覆盖旧 checkpoint。

## 12. 面试官高概率问题与回答

下面的问题按面试追问顺序排列。回答不是背诵稿，而是建议覆盖的技术点。

### A. 项目概览

#### Q1：你这个项目到底解决什么问题？

**答：** 长流程购物 Agent 不只是生成一段文本，而是要在多轮工具交互中搜索、查看商品、选择规格、验证约束并购买。普通 SFT 能学工具格式，但在线 RL 采样会产生 invalid action、all-equal reward 和稀疏终局信号。我实现了从 SFT warm start、GRPO rollout、DAPO dynamic sampling 到 benchmark evaluation 的完整闭环，重点解决 noisy long-horizon trajectory 如何进入稳定 actor update。

#### Q2：你个人做了什么？

**答：** 明确分四层：第一，审计并固定 ShopSimulator、数据 split 和 reward-v3 contract；第二，完成 2x3090 的 Qwen3.5-2B LoRA/FSDP/vLLM 训练路径；第三，修改 veRL adapter，加入 partial-valid dynamic sampling、无效轨迹过滤和 clip-higher；第四，补齐固定分母 Validation-50、Base/SFT/GRPO/DAPO 对照和 paired bootstrap 报告。不要说“从零写了 Qwen3.5”或“发明了 ShopSimulator”。

#### Q3：为什么选择购物 Agent，而不是普通文本分类？

**答：** 购物任务同时包含状态跟踪、工具调用、搜索决策、约束满足、长上下文和终局 reward，比单轮分类更能体现 Agentic RL。每次 action 都改变 environment state，错误会累积，适合展示长流程 RL 的工程难点。

#### Q4：你的创新点是什么？

**答：** 不是提出全新理论，而是把 DAPO-style dynamic sampling 适配到 noisy browser/tool environment：只保留有足够相对信号的有效 peers，遇到 invalid/all-equal group 就 bounded resampling 或跳过更新，再配合 asymmetric clip-higher。同时，我把它做成可恢复、可审计、固定 benchmark 可量化的完整 pipeline。这个创新属于工程方法和系统适配，不应包装成论文级新算法。

#### Q5：为什么不是 HGPO？

**答：** HGPO 的历史分层 credit 在当前购物环境里没有稳定优于 GRPO，Final-200 分支也没有形成可靠 gain；同时它引入了更多信用分配假设。当前主线选择更接近公开实践的 DAPO-style dynamic sampling，问题边界更清楚、实现更容易解释和复现。

#### Q6：你项目的最终结果是什么？

**答：** 固定 Validation-50 上，Base 0%、SFT 58%、GRPO step-25 60%、DAPO step-200 66%。DAPO 相对 SFT +8pp，paired 95% CI `[+2,+16]`；相对 GRPO +6pp，CI `[0,+14]`。但 done/reward-valid 是 90%，低于 SFT，所以我不会说 DAPO 在所有维度上更可靠。

#### Q7：这个结果是不是只因为训练更久？

**答：** 这是合理质疑。step-25 DAPO 与 GRPO 基本持平，step-200 才出现正向差异，因此目前能证明“在更长 budget 下，该配置有效”，不能把全部收益归因于 dynamic sampling。严格的归因需要同预算的 GRPO-200 baseline 和 DAPO ablation，这两个实验是后续最重要的补充。

### B. GRPO 基础

#### Q8：GRPO 是什么？

**答：** GRPO 是 Group Relative Policy Optimization。对同一个 prompt 采样多条 response，用这组 response 的 reward 均值作为 baseline，得到相对优势，再使用 PPO-style clipped ratio 更新 policy。它不需要单独训练 critic，因此对 2x3090 更省显存。

#### Q9：GRPO 和 PPO 的区别？

**答：** PPO 通常需要 policy、old policy、value/critic 和 GAE。GRPO 用同 prompt 的 group reward 构造 baseline，去掉 value model；ratio/clipping 仍然保留。GRPO 的优点是省 critic 成本，缺点是更依赖 group diversity 和 reward quality。

#### Q10：GRPO 的 advantage 怎么算？

**答：** 对 prompt 的 group reward `r_i`，先计算 `mu = mean(r)`，本项目使用 `A_i = r_i - mu` 的 centered relative advantage，并在配置中关闭 `norm_adv_by_std_in_grpo`。然后把 response-level advantage broadcast 到该 response 的 token，进入 clipped policy loss。

#### Q11：为什么同一个 prompt 要采样 4 次？

**答：** 一次 response 只有绝对 reward，没有 prompt 内相对参照。4 次可以估计“哪些行动轨迹相对更好”，形成 group-relative signal；代价是 rollout 成本约为单采样的 4 倍，而且 group 可能全部相同，反而没有有效梯度。

#### Q12：GRPO 为什么不需要 critic？

**答：** critic 的作用是估计 value baseline。GRPO 用同 prompt group 的 reward 均值作为 baseline，牺牲部分时序 value 精度换取显存和实现复杂度下降。对于终局 reward 的 shopping trajectory，这个 trade-off 可接受，但长时序 credit 仍然较弱。

#### Q13：GRPO 是 on-policy 还是 off-policy？

**答：** 训练 rollout 由当前或近似当前 policy 生成，属于 on-policy/near-on-policy policy optimization。PPO ratio 用 old policy log-prob 约束新 policy 不能偏离过远。环境 replay 不是当前主线的一部分。

#### Q14：为什么不用 DQN？

**答：** Action space 是自然语言工具调用和参数组合，离散动作空间巨大，状态和 response 也很长，DQN 不适合直接处理。Policy-gradient 类方法可以直接优化生成策略和 tool call sequence。

#### Q15：为什么不用标准 PPO？

**答：** 标准 PPO 需要 critic，2x3090 资源有限；另外当前 reward 主要是终局 reward，group-relative baseline 更容易先跑通。若后续要做更细粒度 step-level credit，可以加入 value head 或 GAE 做 ablation。

#### Q16：什么是 PPO ratio？

**答：** `rho_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))`。它衡量新旧策略对同一 action token 的概率变化。PPO 用 clip 把 ratio 限制在安全区间，防止单次更新过大造成 policy collapse。

#### Q17：为什么 advantage 为正时上限和下限的作用不同？

**答：** 对正 advantage，过度提高好 action 概率会被 upper clip 截断；对负 advantage，过度降低坏 action 概率会被 lower clip 截断。clip-higher 主要放宽正向学习侧，允许高优势 response 得到更强更新。

#### Q18：你们的 loss 是 sequence mean 还是 token mean？

**答：** 当前 actor 使用 token-mean。长轨迹 token 数多，如果 sequence sum 会让长 response 权重过大；token mean 可以让每个有效 token 的贡献更均衡。但它不能完全消除长轨迹的探索和 credit bias。

#### Q19：为什么配置里没有 KL loss？

**答：** 当前 `use_kl_loss=false`，主要依赖 PPO ratio clipping 和小学习率控制 policy drift。这样减少显存和计算；代价是缺少显式 reference-model KL 约束。若出现 reward hacking 或模型语言能力退化，可以加入 KL coefficient 做稳定性 ablation。

#### Q20：为什么关闭 entropy bonus？

**答：** `entropy_coeff=0`、`calculate_entropy=false`。多轮工具任务的动作本身已经有 rollout temperature 探索，额外 token entropy 会增加计算和噪声。若后续发现 group diversity 不足，可以单独做 entropy coefficient 或 temperature sweep，而不是和 DAPO 同时改。

### C. DAPO 和 dynamic sampling

#### Q21：DAPO 在你项目里具体做了什么？

**答：** 两个可验证机制：一是 asymmetric clip-higher `[0.20, 0.28]`；二是针对无效/恒定 reward group 的 bounded dynamic sampling。每个 prompt 最多尝试 3 个 generation batch，保留至少两个 valid peer，否则跳过 update。

#### Q22：为什么 all-equal reward 不能更新？

**答：** 如果 group 内所有 reward 都一样，centered advantage 全部接近 0，没有 prompt 内偏好方向。强行更新只会引入数值噪声或依赖无意义的 token-level差异，因此跳过比伪造学习信号更合理。

#### Q23：为什么 invalid trajectory 不能直接 reward=0 后训练？

**答：** invalid 可能来自环境 slot、工具解析、action guard 或 reward unverifiable，不一定代表 policy 质量为零。直接设为 0 会把基础设施失败误当成策略偏好，污染 advantage。当前实现区分 `reward_valid`、`sampling_invalid` 和正常 terminal utility，尽量只让可解释 peers 进入 update。

#### Q24：partial-valid sampling 会不会改变 GRPO 理论？

**答：** 会改变 group construction，但不是偷偷改变 reward。它只在 group 中有足够 valid peers 时保留相对信号；invalid group 被过滤或跳过。严格来说这是对原始 whole-group estimator 的工程修正，可能带来 selection bias，因此必须在报告中记录过滤比例、resampling 和 skipped update，而不能声称与标准 GRPO 完全等价。

#### Q25：动态采样是不是“挑 reward 好的样本作弊”？

**答：** 不是按高 reward 排序后只留最好样本，而是先做 validity 和 diversity gate。合法的低 reward trajectory 仍可作为相对比较成员；只过滤不可验证、基础设施无效或没有方差信号的 group。若被追问，应展示 `filtered_groups`、`sampling_invalid_groups` 和 `skipped_updates_total` 日志。

#### Q26：你们最多 resample 几次？

**答：** `max_num_gen_batches=3`。这是 bounded recovery，防止一个 prompt 无限占用 rollout budget。连续跳过达到 `max_consecutive_skipped_updates=10` 时需要触发 watchdog/停止策略，而不是无限生成。

#### Q27：为什么 step-25 没有提升，step-200 有提升？

**答：** 25 updates 对 2B 多模态 Agent 很短，动态 sampler 还在收集有效 group、策略变化幅度有限；step-200 给足了有效 update 累积时间。现有结果支持“长预算更可能显现 gain”，但没有同预算 GRPO-200，因此不能把因果完全归给 DAPO。

#### Q28：DAPO 的缺点是什么？

**答：** 采样成本增加、有效 batch density 下降、selection bias、不同 prompt 获得的 update 次数不均衡，以及 lower done/reward-valid rate。它解决的是 noisy rollout update reliability，不是自动解决 reward design 或 tool protocol。

#### Q29：为什么不继续 HGPO？

**答：** HGPO 的层级 credit 假设和当前工具环境的 reward provenance 没有形成稳定收益，且 Final-200 分支没有验证 superiority。DAPO 的机制更简单、与公开 Agentic RL 实践接近，也更容易在面试中解释每个过滤条件的作用。

#### Q30：如果要证明 DAPO 真正优于 GRPO，还缺什么？

**答：** 必须补 GRPO step-200 同预算 baseline，至少 2-3 个 seed 或 repeated evaluation，并做 dynamic sampling off、clip-high off、partial-valid off 的 ablation。当前 GRPO-25 只能说明 DAPO-200 比短预算 baseline 好，不能单独隔离算法和训练时长。

### D. Reward、环境与 Agent

#### Q31：购物环境的 state/action 是什么？

**答：** state 是用户需求、当前页面 observation、商品列表、页面导航、已选商品/规格和工具可用性；action 是结构化工具调用，例如 search_products、open_product、select_option、next_page、buy_now 或 finish_without_purchase。每次 action 都推进 environment state。

#### Q32：为什么要用工具 guard？

**答：** 自然语言模型可能点击不在当前 observation 的 ASIN 或按钮，直接让环境接受会产生不可复现的非法状态。guard 对当前 observation 中的可用 action 做约束，同时记录 rejection reason，用来区分模型错误和环境错误。

#### Q33：模型如何决定什么时候 buy？

**答：** Agent system prompt 和工具 schema 规定：只有最新 observation 显示 Buy Now，且商品和规格满足用户约束时才能 buy。否则继续检查或搜索；无合适商品时经过最低探索后使用 finish_without_purchase。

#### Q34：为什么 reward 是终局 reward，而不是每步 reward？

**答：** 购买正确性需要看到最终商品、规格和价格，过早给高 reward 容易鼓励模型只做局部动作。终局 reward 更直接对应用户目标，但 credit assignment 更难，这也是 GRPO/DAPO 的主要挑战。

#### Q35：如何防止重复搜索刷 reward？

**答：** environment 记录重复 canonical action、loop、max steps 和 unfinished penalty；Reward-v3 对 repeat_loop 和 max_steps 给负分。评测还报告 average steps、guard rejection 和 status counts，而不只看 purchase success。

#### Q36：Base 为什么是 0%？

**答：** 原始 base 没有学会本项目严格的工具 schema、状态约束和多轮终止 protocol。它不是“Qwen3.5 不会购物”的证明，而是说明 action-only SFT 对 tool protocol acquisition 很关键。

#### Q37：SFT 数据是什么？

**答：** action-only SFT 数据来自同一购物环境的有效工具轨迹，重点教模型按照 schema 搜索、打开、选择和终止，而不是给模型训练一个额外 reward model。SFT merged checkpoint 再作为 GRPO/DAPO 的共同初始化，保证匹配比较。

#### Q38：训练数据和评测数据有没有泄漏？

**答：** launcher 对 RL train、Validation-50 和 Final-200 做 task ID overlap check，并固定 parquet/jsonl hash。面试中应展示 run_manifest 和 split hash；不要只口头说“没有泄漏”。

### E. 模型、LoRA 和多模态

#### Q39：为什么用 Qwen3.5-2B？

**答：** 2B 规模适合两张 24 GiB 3090 做真实多轮 rollout 和 actor update，同时保留 multimodal architecture，便于后续接入图像商品属性。更大模型会提高基础能力，但显存和 rollout latency 迅速上升。

#### Q40：LoRA 是什么？

**答：** LoRA 冻结原始权重，只学习低秩增量：`W' = W + BA`，其中 rank 远小于 hidden dimension。本项目 rank 8、alpha 16、target modules all-linear，减少 trainable parameters 和 optimizer memory。

#### Q41：为什么 LoRA 适合 Agentic RL？

**答：** 在线 rollout 很贵，训练不希望每一步修改完整 2B 模型。LoRA 可以快速更新 policy、降低显存并保留 base prior。缺点是 adapter capacity 有限，可能无法修复深层 reasoning 或视觉 grounding。

#### Q42：为什么 freeze vision tower？

**答：** 当前主 benchmark 没有视觉 observation，更新 vision tower 没有直接监督；冻结它可以显著降低显存、避免对预训练视觉能力造成破坏。视觉能力提升必须在真正含图片的 task 和 ablation 上验证。

#### Q43：这个项目算多模态吗？

**答：** 算“多模态底座上的 Agentic RL pipeline”，不算“已经证明视觉 Agent 提升”。Qwen3.5 保留 vision tower 和 processor，但当前 ShopSimulator 主路径是文本/结构化 observation，vision tower frozen。

#### Q44：如果面试官质疑你没有用图像怎么办？

**答：** 直接承认，并说明这是实验边界。下一步是将商品图像/zoom observation 接入工具轨迹，构造 visual attribute hard gate，做 text-only vs text+image ablation。诚实边界比把文本 benchmark 包装成视觉结果更可信。

#### Q45：LoRA merge 后发生什么？

**答：** FSDP shards 先通过 veRL model merger 导出 standalone checkpoint，再把 LoRA adapter 和 base weight 合并成 BF16 model，供 vLLM evaluator 加载。训练时保存 optimizer/RNG，部署和评测时只需 merged model。

#### Q46：为什么 merge 时有 visual patch embedding missing adapter keys warning？

**答：** vision tower 被 freeze，adapter 中不一定有对应的视觉 LoRA delta；这不是丢失训练更新，而是该模块本来没有被优化。需要检查 merged config、权重加载和 vLLM smoke，而不是把 warning 直接忽略。

### F. 训练系统和并行

#### Q47：FSDP 解决什么问题？

**答：** FSDP 在 data-parallel ranks 间分片 parameters、gradients 和 optimizer state，forward/backward 时按需 all-gather，显著降低每张 3090 的峰值显存。这里 two ranks 对应两张 GPU。

#### Q48：FSDP 和 DDP 的区别？

**答：** DDP 每张卡保留完整模型和 optimizer，梯度 all-reduce；FSDP 进一步分片模型和状态，显存更省但通信和实现复杂度更高。2B multimodal + long context 更适合 FSDP。

#### Q49：为什么 vLLM 和 actor update 要分开？

**答：** vLLM 擅长高吞吐 generation，FSDP actor 擅长训练。系统通过 rollout worker 获取 response/log-prob，再把数据送回 actor update；两者需要合理 GPU memory utilization 和 cache release，避免 KV cache 挤占训练显存。

#### Q50：为什么会出现 GPU duplicate detected 风险？

**答：** Ray 可能给每个 actor 设置不同的 CUDA_VISIBLE_DEVICES，而 NCCL/FSDP rank 仍认为需要两张物理卡，最终多个 rank 映射到同一设备。launcher 显式设置 `CUDA_VISIBLE_DEVICES=0,1` 和 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`，让 rank-to-device 映射一致。

#### Q51：为什么使用 gradient checkpointing？

**答：** 不保存每层完整 activation，反向时重算 forward，以计算时间换显存，适合 long-context multi-turn trajectory。它会降低 throughput，但让 24 GiB GPU 能完成真实训练。

#### Q52：训练时遇到 ShopSimulator slot exhausted 怎么办？

**答：** 这是环境服务资源生命周期问题，不应该当作低 reward 直接训练。先释放/重启 ShopSimulator，检查 session release，再从最近完整 checkpoint resume。实际 run 中遇到过此问题，最终从 step-100 恢复到 step-200，旧 checkpoint 保留。

#### Q53：如何判断 NaN/OOM 和环境错误？

**答：** 看三层日志：GPU memory/worker stderr；actor loss、grad_norm、KL 是否 finite；environment reward version/session error。OOM 通常伴随 CUDA allocation，环境错误则是 ShopEnvironmentError 或 no available environment。两者处理方式完全不同。

#### Q54：为什么不要把训练 reward 当最终结果？

**答：** 训练 reward 是 rollout policy、采样温度、动态过滤和当前 train batch 的局部统计；可能被 selection、task difficulty 和 reward shaping 影响。最终必须用固定 task、greedy decoding、同一 evaluator 重新跑 benchmark。

### G. 评测和统计八股

#### Q55：strict success 的精确定义是什么？

**答：** `reward_version == shopsimulator-reward-v3`、trajectory status/done/terminal done/over 都为 true、reward_type 是 gold_purchase、reward_valid 和 purchase_success 为 true、termination_reason 是 gold_purchase。合法替代不计入 strict success。

#### Q56：purchase success 和 strict success 有什么差异？

**答：** purchase success 允许 gold_purchase 或 valid_alternative_purchase；strict success 只允许精确目标商品的 gold_purchase。本次两者数字相同是数据结果，不是定义相同。

#### Q57：为什么 fixed denominator 很重要？

**答：** invalid/unfinished task 不能被从分母删掉，否则模型只要让难任务报错就能提高“完成任务成功率”。固定 50 task denominator 把 invalid action 作为失败并另外报告 done/reward-valid caveat。

#### Q58：paired bootstrap 和普通 bootstrap 有什么区别？

**答：** 普通 bootstrap 可能独立重采样两组样本，忽略同一 task 的配对关系；paired bootstrap 重采样 task ID，并对每个 task 先计算 candidate-baseline delta，能控制 task difficulty。

#### Q59：CI 包含 0 说明什么？

**答：** 在当前有限样本和 bootstrap 设定下，不能排除真实 delta 为 0 或负数。GRPO->DAPO strict CI `[0,+14]` 是 borderline positive，reward CI 略跨 0，所以不能说已经统计显著全面优于 GRPO。

#### Q60：为什么只评 50 个任务？

**答：** 这是 fixed validation gate，用于低算力 checkpoint/方法比较，不是最终论文规模。为了简历可信，还要补 untouched held-out split、多 seed 或 repeated evaluation。

#### Q61：Base 0% 会不会是评测 bug？

**答：** 要检查 raw trajectories、tool schema、HTTP response、status counts 和 reward version。Base 主要表现为 34 个 invalid-action-limit 和大量 guard rejection；SFT 在同一 evaluator 达到 58%，说明环境和 evaluator 可工作，但 base 的 tool protocol acquisition 失败。仍然应该在面试中承认 base floor 可能受 system prompt/tool-format mismatch 影响。

#### Q62：你做过哪些 sanity check？

**答：** 固定 task IDs、重复 baseline model、检查 missing/unexpected IDs、验证 reward contract、greedy temperature 0、检查 vLLM `/v1/models`、JSON trajectory 完整性、checkpoint model load、FSDP merge smoke、GPU memory 和 finite gradients。

### H. 结果、限制与反事实追问

#### Q63：DAPO 的 +8pp 能完全归因于算法吗？

**答：** 不能完全归因。DAPO-200 和 GRPO-25 不同训练预算；当前最稳妥结论是“DAPO 配置在 200 updates 下相对 SFT 提升，且相对短预算 GRPO 正向”。要隔离算法，必须补 GRPO-200 同预算 baseline。

#### Q64：为什么 DAPO done rate 低于 SFT？

**答：** 可能来自更长探索、不同 tool routing 和 invalid-action distribution。当前 DAPO 的 guard rejection 和 invalid-action-limit 更多，说明动态采样改善了部分 reward learning，但还没有完全改善 protocol robustness。

#### Q65：你最不满意的地方是什么？

**答：** 第一是 GRPO-200 同预算对照缺失；第二是 Validation-50 规模小；第三是当前主 benchmark 没有真正视觉输入；第四是 DAPO done/reward-valid 下降。把这些主动说出来比等面试官抓住更好。

#### Q66：如果结果没有提升，项目还成立吗？

**答：** 成立，但结论会从 algorithm gain 改为 engineering reliability and evaluation pipeline。项目的价值还包括可恢复 long-horizon rollout、invalid group filtering、FSDP/LoRA deployment 和 fixed-denominator measurement。不能为了让简历好看而改指标。

#### Q67：你如何排除 SFT 数据本身造成结果？

**答：** 用 Base->SFT->GRPO/DAPO chain 展示 SFT 的能力增益，并保持 GRPO/DAPO 使用同一 merged SFT initializer。若要排除更多因素，需要 Base+RL、SFT-only 多 seed、不同 SFT data size ablation。

#### Q68：如果面试官说 +8pp 只是 4 个 task 的偶然 win？

**答：** 当前 paired transition 是 4 wins、0 losses、29 both-success、17 both-failure；CI `[+2,+16]`。这说明固定 50 split 上有可观察 gain，但样本仍小，应该补更大 held-out 和多 seed，不能声称普遍化。

### I. 代码、调试和复现

#### Q69：入口脚本在哪？

**答：** `scripts/run_shopping_rl_2x3090.py`。它做 preflight、runtime/version/hash 校验，构造 veRL overrides，设置 GPU/Ray/environment variables，并保存 run_manifest。

#### Q70：如何一条命令查看 DAPO 配置？

```bash
third_party/shopping-grpo-longhorizon/.venv/bin/python \
  scripts/run_shopping_rl_2x3090.py --method dapo --dry-run
```

#### Q71：怎么复现最终评测？

```bash
DAPO_MODEL=third_party/shopping-grpo-longhorizon/outputs/models/dapo-qwen35-200-merged \
DAPO_LABEL=dapo_step200 \
bash scripts/evaluate_shopping_dapo_validation50.sh \
  artifacts/evaluations/shopping_dapo_validation50_step200
```

基础模型单独评测：

```bash
bash scripts/evaluate_shopping_base_validation50.sh \
  artifacts/evaluations/shopping_base_validation50
```

#### Q72：如何确认数据没有被改？

**答：** launcher 对 `train.parquet`、`validation.parquet` 和 `tasks.jsonl` 做 SHA-256 hash；同时检查 train/validation/Final-200 task ID 不重叠。评测 benchmark 使用固定 `validation.jsonl`。

#### Q73：为什么要把 upstream 放在 third_party？

**答：** 保留上游边界和 commit provenance，避免把参考项目冒充自研。项目自己的 launcher、dynamic sampling patch、evaluation/report 和 integration tests 在根项目；上游环境只做必要适配。

#### Q74：如果 vLLM 输出和训练 rollout 不一致怎么办？

**答：** 固定 model export、tokenizer/processor、tool parser、temperature/top-p、max tokens；比较 raw prompt、rendered chat、tool call parsing 和 context compaction。训练 rollout 采用 temperature 0.7 做探索，最终 evaluator 使用 temperature 0 做确定性比较，所以输出不应逐条相同，但协议和模型必须一致。

### J. Transformer/LLM/PEFT 八股

#### Q75：Transformer 为什么适合工具 Agent？

**答：** self-attention 能把当前用户需求、历史 observation、工具结果和已选规格放入同一上下文，生成下一步 action。缺点是长历史的 quadratic attention、上下文污染和错误累积，所以需要 compaction、structured observation 和 action guard。

#### Q76：attention 的复杂度是多少？

**答：** 标准 self-attention 对序列长度 `L` 是 `O(L^2 * d)` 计算和 `O(L^2)` attention matrix；推理阶段 KV cache 将每个新 token 的历史访问变成近似线性增长，但 cache memory 仍随 `L` 和 layer/head 增加。

#### Q77：为什么长 tool observation 会伤害模型？

**答：** 大量商品列表和页面文本会稀释用户约束、超过 context budget、增加错误 ASIN 选择概率。项目使用 structured observation、detail/generic token budget、middle truncation 和 context compaction，同时记录 truncation/guard failures。

#### Q78：LoRA 的参数量为什么少？

**答：** 对 `d_out x d_in` 的 full update，LoRA 只训练 `d_out x r + r x d_in`，当 `r=8` 远小于 hidden dimension 时参数和 optimizer state 大幅下降。

#### Q79：BF16 和 FP16 的区别？

**答：** BF16 exponent bits 与 FP32 接近，动态范围更大，训练更不容易 overflow；FP16 mantissa 更细但范围较小，通常需要 loss scaling。3090 上本项目用 BF16 merged serving，并结合 gradient checkpoint/offload 控制显存。

#### Q80：temperature 和 top-p 对 RL 有什么作用？

**答：** training rollout temperature 0.7/top-p 0.9 提供 exploration；evaluation temperature 0/top-p 1 用于 deterministic comparison。不能把探索期 reward 和 greedy benchmark reward 混为一谈。

### K. RLHF/Agentic RL 八股

#### Q81：SFT、RLHF、RLAIF、Agentic RL 的关系？

**答：** SFT 用 demonstrations 学行为格式；RLHF 用 preference/reward 优化偏好；RLAIF 用 AI feedback 代替人工偏好；Agentic RL 把 action sequence 放入可交互环境，由环境终局或 verifier 给 reward。当前项目属于 environment-verifiable Agentic RL。

#### Q82：exploration 和 exploitation 怎么平衡？

**答：** training temperature 负责采样探索，GRPO group diversity 提供相对比较，DAPO dynamic sampling 过滤无信号 group；clip-higher 允许高优势 trajectory 更快学习。过度探索会增加 invalid loops，过度 exploitation 会让 group 全部同 reward。

#### Q83：credit assignment 为什么难？

**答：** reward 在最终 purchase 才知道，但错误可能发生在搜索 query、翻页、打开商品或选择规格的早期。GRPO 把终局 advantage broadcast 到 response token，粒度粗；更细的 event/step credit、critic 或 hindsight relabeling 是后续方向。

#### Q84：reward model 和 environment verifier 有什么差异？

**答：** reward model 是学习出来的近似偏好函数，可能被 policy exploit；environment verifier 使用商品目录、约束和购买状态的规则，解释性和可复现性更强。本项目优先使用 ShopSimulator Reward-v3，避免再引入一个 2B 资源受限的 reward model。

#### Q85：什么是 off-policy correction？

**答：** 当数据由旧 policy 产生而用新 policy 更新时，importance ratio/behavior policy log-prob 用来修正分布偏差。PPO clipping 是一种受控近似；本项目 rollout 与 actor update 紧邻，且没有长期 replay buffer，因此偏离较小。

#### Q86：GRPU 是什么？你项目里是不是 GRPU？

**答：** 本项目使用的是 **GRPO（Group Relative Policy Optimization）**，不是 GRPU。口述时“GRPO”容易说错；如果面试官或自己提到 GRPU，应先确认是否只是笔误。项目里的 group-relative advantage、PPO ratio 和 clipped objective 都对应 GRPO。不要为了顺着错误缩写临时编一个算法。

## 13. 面试官可能要求的 ablation

如果被问“你如何证明每个模块有效”，按下面优先级回答：

1. **GRPO-200 vs DAPO-200**：隔离训练 budget 与算法。
2. **DAPO clip-high off**：`[0.20,0.20]` vs `[0.20,0.28]`。
3. **partial-valid off**：whole-group vs partial-valid，比较 effective update density 和 strict success。
4. **dynamic sampling off**：固定相同 rollout budget，观察 skipped update、reward-valid、done rate。
5. **SFT data-size ablation**：Base、少量 SFT、完整 SFT，验证 protocol acquisition。
6. **rollout group size**：`G=2/4/8`，观察 group variance、成本和 gain。
7. **seed/repeated evaluation**：至少 3 seed 或每个 task 多次 sampled rollout。
8. **vision ablation**：text-only、image-enabled、frozen vision、LoRA vision adapter。

回答原则：一次只改一个变量，固定 data hash、task IDs、model family、GPU budget 和 evaluator protocol。

## 14. 项目目前的诚实结论

### 已经证明

- 能在 2x3090 上稳定完成 Qwen3.5-2B shopping Agent 的 SFT/online RL/导出/评测闭环。
- SFT 对工具协议获取有巨大作用：Base 0% -> SFT 58%。
- DAPO step-200 在固定 Validation-50 上相对 SFT 有 +8pp strict success。
- 动态采样可以识别并过滤 all-equal、invalid reward group，训练过程能恢复和记录。

### 还没有证明

- DAPO 在同预算 GRPO-200 上的纯算法优势。
- 视觉商品理解或多模态 grounding 提升。
- 大规模 benchmark 上的泛化或 SOTA。
- done/reward-valid 等可靠性指标全面提升。

## 15. 5 分钟项目讲解顺序

1. 用一句话说明长流程购物 Agent 和目标。
2. 说清 Base/SFT/GRPO/DAPO 四阶段数字。
3. 解释环境 reward-v3 和 strict success，证明指标不是训练 reward。
4. 画 rollout -> group -> dynamic filtering -> actor update 流程。
5. 推导 GRPO group advantage 和 PPO ratio。
6. 解释 partial-valid 为什么适合 noisy tool environment。
7. 说明 2x3090 的 LoRA/FSDP/vLLM 工程取舍。
8. 主动说 DAPO done rate 下降、GRPO-200 缺失和视觉 benchmark 未验证。
9. 给出下一步 ablation：同预算 GRPO-200、clip/partial-valid ablation、held-out split。

## 16. 最后检查清单

- [ ] 能在 30 秒内说清项目问题、方法、结果、限制。
- [ ] 能写出 `A_i = r_i - mean(group rewards)` 和 PPO ratio。
- [ ] 能解释为什么 all-equal group 跳过 update。
- [ ] 能解释 strict success 与 purchase success 的区别。
- [ ] 能说出 Base/SFT/GRPO/DAPO 四个准确数字。
- [ ] 能说出 DAPO 的 +8pp CI `[+2,+16]`，而不是只说 66%。
- [ ] 能主动承认本次主 benchmark 是文本工具购物，不是视觉能力测评。
- [ ] 能指出 GRPO-200 是当前最关键的缺失对照。
- [ ] 能解释 FSDP、LoRA、gradient checkpointing、vLLM 各自解决什么问题。
- [ ] 能展示 `run_manifest`、checkpoint、summary.json 和 paired comparison JSON。
