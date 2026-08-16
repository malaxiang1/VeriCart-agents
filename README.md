# VeriCart-Agent

基于 Qwen3.5-2B 的长链路购物智能体强化学习项目，面向双卡 RTX 3090 环境。

本仓库提供可复现的项目代码、训练配置、评测协议、轻量化指标结果和面试说明。
模型权重、数据集、完整轨迹、虚拟环境和运行时下载内容均不包含在仓库中。

## 项目解决的问题

目标是让购物策略把自然语言需求转化为可验证的多步工具流程：

```text
理解约束 -> 搜索商品 -> 查看详情 -> 选择规格
-> 核验价格/属性 -> 购买或安全终止
```

项目重点研究长链路工具轨迹中存在非法动作、终止奖励不可验证、奖励全部相同、
缺少相对学习信号时，如何保持 Agentic RL 更新的有效性和稳定性。

## 实验结果

所有策略均在同一组固定的 50-task ShopSimulator v2.1 validation split 上评测，
使用固定的 `shopsimulator-reward-v3` reward contract。

| 策略 | 严格购买成功率 | 平均 reward-v3 | 正常结束率 | reward 有效率 |
|---|---:|---:|---:|---:|
| 原始 Qwen3.5-2B | 0% | -0.15513 | 28% | 26% |
| Action-only SFT | 58% | 0.39085 | 96% | 96% |
| 原生 GRPO，step 25 | 60% | 0.43846 | 96% | 94% |
| DAPO-style，step 200 | **66%** | **0.51838** | 90% | 90% |

DAPO step-200 相较 SFT 的严格购买成功率提升 `+8` 个百分点，配对 bootstrap
95% CI 为 `[+2,+16]`；相较短步数 GRPO 基线提升 `+6` 个百分点，CI 为 `[0,+14]`。
平均 reward-v3 相较 SFT 提升 32.63%。同时，DAPO 的正常结束率和 reward 有效率较低，
因此本项目不声称在所有可靠性维度上都更优。

## 方法

- 使用具备多模态架构的 Qwen3.5-2B 作为底座；本项目的主 benchmark 使用文本/结构化工具观测，视觉塔保持冻结。
- 使用 Action-only LoRA SFT 作为训练 warm start。
- 每个 prompt 采样四条多轮工具轨迹。
- 使用不依赖 Critic 的原生 GRPO 组内相对优势。
- 对 DAPO-style `partial_valid` 动态采样进行场景化改造：过滤非法/奖励全部相同的 group，有限重采样，并跳过缺少有效相对信号的更新。
- 使用 DAPO-style clip-higher 配置：`clip_ratio_low=0.20`，`clip_ratio_high=0.28`。
- 使用 token-mean actor loss、LoRA rank 8、FSDP 分片、梯度检查点、CPU 参数/优化器 offload 和 vLLM rollout。

本项目是面向购物工具环境的 DAPO-style 工程适配，不声称完整复现 DAPO 论文的全部细节。
当前主 benchmark 验证的是文本/结构化购物交互，不宣称视觉 grounding 能力提升。

## 仓库结构

```text
configs/       双卡 RL 和 AgentLoop 配置
scripts/       训练、评测、导出和结果对比入口
groundedvision/配对 benchmark 对比和 bootstrap 工具
tests/         评测和 launcher 单元测试
patches/       固定上游 shopping 和 veRL runtime patch
results/       轻量化 summary JSON 和配对比较 JSON，不含原始轨迹
docs/          实验报告、结论审计和面试说明
```

## 复现流程

### 1. 准备固定版本的上游环境

运行时依赖公开上游项目的固定 commit
`c3c178595eea835c18ba4515d553025014e52656`：

```bash
git clone https://github.com/YYHDBL/shopping-grpo-longhorizon.git \
  third_party/shopping-grpo-longhorizon
cd third_party/shopping-grpo-longhorizon
git checkout c3c178595eea835c18ba4515d553025014e52656
```

请按照上游项目的说明安装运行环境。经过测试的版本包括 veRL 0.8.0、vLLM 0.18.0、
PyTorch 2.10.0+cu128、Transformers 5.15.0.dev0 和 FlashInfer 0.6.6。

### 2. 应用项目 patch

在上游仓库根目录应用 shopping source patch：

```bash
git apply ../../patches/shopping-grpo-c3c1785.patch
```

在环境的 veRL site-packages 根目录应用 runtime patch。该 patch 基于原始 veRL 0.8.0
的 `ray_trainer.py` 生成：

```bash
cd .venv/lib/python3.12/site-packages
patch -p1 < ../../../../../../patches/verl-0.8.0-vericart.patch
```

Python 小版本不同会导致相对路径不同，可使用下面的命令定位安装位置：

```bash
python -c "import verl; print(verl.__file__)"
```

### 3. 运行预检或训练

Qwen3.5-2B 权重不包含在仓库中。运行 SFT 时请将 `MODEL` 设置为本地 checkpoint，
运行 RL 时传入合并后的 SFT checkpoint。双卡训练还需要启动 ShopSimulator 服务：
`http://127.0.0.1:5700`。

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

### 4. 运行固定评测

```bash
BASE_MODEL=/path/to/Qwen3.5-2B \
  bash scripts/evaluate_shopping_base_validation50.sh \
  results/validation50/base

DAPO_MODEL=/path/to/dapo-qwen35-200-merged \
DAPO_LABEL=dapo_step200 \
bash scripts/evaluate_shopping_dapo_validation50.sh \
  results/validation50_step200
```

评测使用 greedy decoding、最多 35 个环境 turn、带上下文压缩的 24576-token evaluator
context，并对所有模型使用相同的 50 个 task ID。

## 结果与证据

- [Step-200 实验报告](docs/shopping_dapo_validation50_step200_report.md)
- [结论审计](docs/shopping_dapo_step200_claim_validation.md)
- [面试说明](docs/interview_project_guide.md)
- [轻量化 validation 结果](results/validation50/)

## 范围与限制

- ShopSimulator v2.1 是开源 benchmark protocol，不是 WebShop 官方 leaderboard。
- 当前主 benchmark 是文本/结构化工具交互，不能证明视觉 grounding 或多模态准确率提升。
- Validation-50 适合做可复现的工程对比，不足以支持广泛的 SOTA 结论。
- 缺少同预算 GRPO-200 对照，这是区分 DAPO 机制与训练时长影响的主要待补实验。
- DAPO 提高了成功率，但当前正常结束率和 reward 有效率下降，后续需要继续修复协议鲁棒性。

## 致谢与第三方说明

发布前请阅读 [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md)。上游仓库作为固定版本依赖，
本仓库不将其代码标记为原创项目代码。
