<div align="center">

# DeepSleep

**睡眠健康领域轻量级 MoE 大语言模型**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**~200M 参数 MoE** | 中英双语 | 睡眠健康领域 | 单卡可训练

[快速开始](#-快速开始) · [模型架构](#-模型架构) · [训练流程](#-训练流程) · [实验分析](#-实验分析) · [项目结构](#-项目结构)

</div>

---

## 项目简介

DeepSleep 是一个从零开始构建的睡眠健康领域 MoE 大语言模型，灵感来自 [MiniMind](https://github.com/jingyaogong/minimind) 和 [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-moe/)。项目完整实现了大模型的全部流程：**Tokenizer 训练 → 预训练 → CPT → SFT → DPO → Benchmark 评估 → 统计分析**，所有代码开源可复现。

**核心特点：**

- **MoE 架构** — 8 层全 MoE，~199M 总参数 / ~65M 活跃参数，softmax 路由
- **主流组件** — GQA + RoPE + RMSNorm + SwiGLU + Flash Attention
- **中英双语** — 7200 BPE 词表，从 CCI4.0-HQ 语料训练
- **小曦人格** — 温暖有趣的睡眠健康伙伴，10000 条 SFT + 1965 对 DPO 数据
- **单卡可训练** — NVIDIA A10 (24GB) 即可完成全流程
- **一键启动** — YAML 配置 + Shell 脚本，开箱即用
- **完整实验分析** — 2² 全因子 ANOVA + Benchmark 多因素方差分析 + 生成质量评估

---

## 模型架构

```
DeepSleepForCausalLM (~199M params)
├── Embedding (vocab=7200, d_model=768, tied with lm_head)
├── 8 MoE Layers
│   ├── DeepSleepAttention (GQA: 8Q/4KV heads, head_dim=96, RoPE, Flash/SDPA)
│   └── DeepSleepMoE (8 routed experts, top_k=2, SwiGLU, intermediate=1216)
├── Final RMSNorm
└── LM Head (tied, no bias)

Total: ~199M | Active per token: ~65M | Utilization: 32.4%
```

| 超参数 | 值 |
|--------|-----|
| d_model | 768 |
| n_layers | 8 |
| n_heads / n_kv_heads | 8 / 4 (GQA) |
| head_dim | 96 |
| num_routed_experts | 8 |
| top_k | 2 |
| moe_intermediate_size | 1216 |
| vocab_size | 7,200 |
| max_position_embeddings | 8,192 |

**主流组件：** GQA · RoPE · RMSNorm · SwiGLU · Flash Attention (SDPA) · Pre-Norm

---

## 快速开始

### 1. 环境配置

```bash
git clone https://github.com/L-0915/deepsleep-replace.git
cd deepsleep-replace

conda create -n deepsleep python=3.10 -y
conda activate deepsleep
pip install -r requirements.txt
```

### 2. 下载模型权重

模型权重托管在 [ModelScope (魔搭)](https://www.modelscope.cn/models/shephub/deepsleep)，按以下目录结构放置：

```
deepsleep-replace/
├── out/
│   ├── pretrain/final/model.pth          # deepsleep-pretrain.pth
│   ├── cpt/final/model.pth               # deepsleep-cpt.pth
│   ├── sft/final_model.pth               # deepsleep-sft.pth
│   ├── sft_qwen/final_model/model.safetensors  # qwen-sft.safetensors
│   ├── ds_b0.1_hf/model.safetensors      # deepsleep-dpo-b0.1-hf.safetensors
│   └── ds_b0.5_hf/model.safetensors      # deepsleep-dpo-b0.5-hf.safetensors
├── dpo_exp/                               # DPO 实验模型（按需下载）
│   ├── ds_b0.1_s42/final_model.pth
│   ├── ds_b0.5_s42/final_model.pth
│   ├── qwen_b0.1_s42/final_model/model.safetensors
│   ├── qwen_b0.5_s42/final_model/model.safetensors
│   └── ...（共12个DPO模型）
```

**方式一：使用 modelscope SDK 下载**

```bash
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('shephub/deepsleep', cache_dir='./model_weights')
"
# 然后将下载的文件按上述目录结构复制
```

**方式二：手动下载**

访问 https://www.modelscope.cn/models/shephub/deepsleep/files 逐个下载，放到对应目录。

> **最少只需下载** `deepsleep-sft.pth` 即可运行 Gradio 演示。如需完整 4 模型对话，还需下载 `ds_b0.1_hf/`、`ds_b0.5_hf/`、`qwen_b0.1_s42/final_model/`、`qwen_b0.5_s42/final_model/`。

### 3. 使用方式

#### 方式 A：Gradio 简易对话（推荐快速体验）

```bash
python app.py --model out/sft/final_model.pth
# 浏览器打开 http://localhost:6006
```

#### 方式 B：React + FastAPI 全功能对话（4模型切换 + 流式输出）

**后端启动：**

```bash
# 安装后端额外依赖
pip install fastapi uvicorn sse-starlette

# 修改 server.py 中 MODEL_CONFIGS 的 path 为你的模型路径
# 然后启动后端
python server.py --host 0.0.0.0 --port 7860
```

**前端启动：**

```bash
cd web

# 安装依赖
npm install

# 开发模式
npm run dev
# 浏览器打开 http://localhost:5173

# 或构建生产版本
npm run build
# 构建产物在 web/dist/，启动 server.py 后会自动托管前端
```

#### 方式 C：命令行推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 DeepSleep HF 格式模型
model_path = "out/ds_b0.5_hf/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
model.cuda().eval()

messages = [
    {"role": "system", "content": "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。"},
    {"role": "user", "content": "失眠了怎么办？"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### 4. 训练 Tokenizer

```bash
# 从 CCI4.0-HQ 流式加载中英文语料训练 BPE tokenizer
python scripts/train_tokenizer.py --use_cci4 --cci4_max_docs 300000
```

### 3. 训练模型

```bash
# 方式一：一键启动（推荐）
bash scripts/run/run_all.sh          # 全流程：pretrain → CPT → SFT → DPO

# 方式二：分阶段运行
bash scripts/run/run_pretrain.sh     # 预训练（流式加载 CCI4.0-HQ）
bash scripts/run/run_cpt.sh          # CPT（睡眠领域语料）
bash scripts/run/run_sft.sh          # SFT 微调（小曦人格数据）
bash scripts/run/run_dpo.sh          # DPO 对齐

# 方式三：用 YAML 配置文件
python trainer/train_pretrain.py --config configs/pretrain.yaml --tokenizer_path checkpoints/tokenizer
python trainer/train_sft.py --config configs/sft.yaml
python trainer/train_dpo.py --config configs/dpo.yaml
```

### 4. Benchmark 评估

```bash
# 使用 lm-evaluation-harness 评估 5 个标准 benchmark
bash scripts/eval/run_benchmark.sh

# 生成雷达图 + CSV 汇总表
python scripts/eval/plot_radar.py
```

### 5. Web 演示

```bash
python app.py --model out/sft/final_model.pth
```

---

## 训练流程

```
Stage 1: Pretrain (流式加载 CCI4.0-HQ)
├── 数据: CCI4.0-HQ (中英文混合, HuggingFace streaming)
├── 结果: 13K steps, loss 1.77, PPL 5.85
└── 输出: out/pretrain/final/model.pth

Stage 2: CPT (睡眠领域继续预训练)
├── 数据: 睡眠健康领域语料
├── 结果: 2K steps, loss 1.14, PPL 3.11
└── 输出: out/cpt/final/model.pth

Stage 3: SFT (小曦人格微调)
├── 数据: 10000 条 ChatML 对话 (6类别)
├   ├── 专业诊断CoT 2500 | 知心安慰 2500 | 趣味科普 1500
├   └── 睡前引导 1000 | 拟人分享 1000 | 个性化互动 1500
├── DeepSleep: 1565 steps, loss 3.62
├── Qwen: 3750 steps
└── 输出: out/sft/final_model.pth, out/sft_qwen/final_model/

Stage 4: DPO (偏好对齐, 2²全因子实验)
├── 数据: 1965 对偏好对比 (6类别)
├── 实验设计: 模型架构(DeepSleep/Qwen) × Beta(0.1/0.5) × 3重复 = 12 runs
├── DeepSleep: β=0.1 loss~0.04, β=0.5 loss~0.0006, accuracy 100%
├── Qwen: β=0.1/0.5 均收敛, accuracy 100%
└── 输出: /root/blockdata/dpo_exp/{ds,qwen}_b*/final_model.*
```

---

## 实验分析

### 2² 全因子 ANOVA

对 DPO 训练结果进行 2² 全因子方差分析，研究**模型架构** (DeepSleep MoE vs Qwen Dense) 和 **DPO Beta** (0.1 vs 0.5) 两个因素对对齐效果的影响。

**关键发现：**
- 模型架构和 DPO Beta 的主效应和交互效应均显著 (p < 0.05)
- DeepSleep MoE 从 β=0.1 到 β=0.5 的 loss 下降幅度远大于 Qwen Dense
- 说明稀疏 MoE 模型从更强的对齐信号中获益更多

**分析脚本：** `scripts/analysis/analyze_factorial.py`
**输出图表：**
- `fig1_training_curves.png` — 4 组训练曲线 (12 runs)
- `fig2_main_effects.png` — 主效应图
- `fig3_interaction.png` — 交互效应图
- `fig4_residuals_*.png` — 残差诊断
- `fig5_boxplots.png` — 箱线图
- `fig6_pareto.png` — Pareto 图

### Benchmark 多因素方差分析

对 8 个模型在 5 个 benchmark 上的 acc_norm 分数进行无重复双因素 ANOVA (Model × Benchmark)。

**实验设计：**
- 因子 A: Model (8 水平) — DeepSleep-Base/DPO(β=0.1/0.5), MiniMind-3, Medical-GPT2, Qwen2.5-Base/DPO(β=0.1/0.5)
- 因子 B: Benchmark (5 水平) — PubMedQA, MedQA, ARC-Easy, PIQA, OpenBookQA
- 设计类型: 随机化完全区组设计 (每格 1 个观测值)
- 交互效应 = 误差项 (无重复时的标准处理)

**分析脚本：** `scripts/analysis/plot_benchmark_anova.py`
**输出图表：** `fig_benchmark_anova_barplot.png/pdf`

### 生成质量评估

使用 30 条统一测试 prompt，由 DeepSeek V4 对 4 个 DPO 模型的回复按 10 个维度打分 (1-10 分)。

**评估维度：** 专业准确性 / 安全合规性 / 人格一致性 / 实用可操作性 / 同理心与关怀 / 思考深度 / 语言自然度 / 知识广度 / 个性化程度 / 回复完整性

**分析脚本：** `scripts/evaluation/eval_quality.py`
**输出图表：** `fig_quality_anova.png`

---

## Benchmark 结果

8 个模型在 5 个标准 benchmark 上的 acc_norm 分数 (%):

| Model | PubMedQA | MedQA | ARC-Easy | PIQA | OpenBookQA |
|-------|----------|-------|----------|------|------------|
| DeepSleep-Base | 20.00 | 27.02 | 31.02 | 55.93 | 26.80 |
| DeepSleep-DPO(β=0.1) | 37.00 | 22.39 | 31.94 | 54.90 | 27.20 |
| DeepSleep-DPO(β=0.5) | 36.00 | 22.31 | 31.90 | 55.11 | 27.20 |
| MiniMind-3 | 49.00 | 26.94 | 28.49 | 50.65 | 23.60 |
| Medical-GPT2 | 51.00 | 27.73 | 28.32 | 51.25 | 29.00 |
| **Qwen2.5-Base** | **57.00** | **35.04** | **58.54** | **69.97** | **35.20** |
| Qwen2.5-DPO(β=0.1) | 60.00 | 36.53 | 56.86 | 69.53 | 33.20 |
| Qwen2.5-DPO(β=0.5) | 59.00 | 36.76 | 57.28 | 69.70 | 33.80 |

> 评估工具: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.13

---

## 小曦人格

**星辰曦（小曦）** 是 DeepSleep 的 AI 人格，定位为温暖、有趣的睡眠健康伙伴。

| 能力 | 数据量 | 示例 |
|------|--------|------|
| 专业诊断 (CoT) | 2500 条 | 症状分析 → 推理 → 建议 |
| 知心安慰 | 2500 条 | 共情 + 实用建议 |
| 趣味科普 | 1500 条 | 睡眠冷知识 + 比喻 |
| 睡前引导 | 1000 条 | 呼吸放松、冥想脚本 |
| 拟人分享 | 1000 条 | 小曦的生活小故事 |
| 个性化互动 | 1500 条 | 记住用户、主动回访 |

---

## 项目结构

```
deepsleep/
├── model/
│   └── model_deepsleep.py          # 完整模型: Config, Attention, MoE, CausalLM
├── dataset/
│   ├── lm_dataset.py               # PretrainDataset, SFTDataset, DPODataset
│   ├── streaming_dataset.py        # CCI4PretrainDataset (流式加载)
│   └── medical_qa_dataset.py       # PubMedQA 数据集
├── trainer/
│   ├── train_pretrain.py           # 预训练 (HuggingFace Trainer, MoE-aware)
│   ├── train_sft.py                # SFT 微调
│   ├── train_dpo.py                # DeepSleep DPO 对齐
│   ├── train_dpo_qwen.py           # Qwen DPO 对齐 (含内存优化)
│   └── trainer_utils.py            # 共享工具函数
├── configs/                        # 训练配置 (YAML)
│   ├── config_utils.py             # YAML → argparse 加载器
│   ├── pretrain.yaml / cpt.yaml    # 预训练/CPT 配置
│   ├── sft.yaml / sft_qwen.yaml    # SFT 配置
│   └── dpo.yaml / dpo_qwen.yaml    # DPO 配置
├── scripts/
│   ├── run/                        # 一键启动脚本
│   ├── eval/                       # Benchmark 评估工具
│   │   ├── run_benchmark.sh        # lm-eval 8模型×5 benchmark
│   │   ├── convert_to_hf.py        # DeepSleep .pth → HF 格式
│   │   ├── merge_results.py        # 合并增量评估结果
│   │   └── plot_radar.py           # 雷达图 + CSV 汇总表
│   ├── analysis/                   # 统计分析脚本
│   │   ├── analyze_factorial.py    # 2²全因子 ANOVA (9张图+报告)
│   │   └── plot_benchmark_anova.py # Benchmark 双因素 ANOVA + 柱状图
│   ├── evaluation/                 # 生成质量评估
│   │   └── eval_quality.py         # DeepSeek V4 10维度打分 + ANOVA
│   ├── generate_xiaoxi_all.py      # 小曦 SFT 数据生成
│   ├── generate_xiaoxi_dpo.py      # 小曦 DPO 数据生成
│   ├── train_tokenizer.py          # BPE 分词器训练
│   └── prepare_deepsleep_data.py   # 预训练数据工具
├── data/
│   ├── sft/xiaoxi/                 # SFT 数据 (10000条 ChatML)
│   ├── dpo/                        # DPO 数据 (1965对偏好对比)
│   └── eval/                       # 评估结果 + PubMedQA 本地数据
├── docs/
│   ├── figures/                    # 所有实验图表 (300DPI)
│   ├── experiment-design-plan.md   # 实验设计计划
│   └── analysis_report.md          # ANOVA 分析报告
├── out/                            # 训练产出 (模型权重+日志)
├── app.py                          # Gradio Web UI
├── Makefile
├── requirements.txt
└── LICENSE
```

---

## 训练产出

| 模型 | 路径 | 关键指标 |
|------|------|----------|
| Pretrain | `out/pretrain/final/` | 13K steps, loss 1.77, PPL 5.85 |
| CPT | `out/cpt/final/` | 2K steps, loss 1.14, PPL 3.11 |
| SFT (DeepSleep) | `out/sft/` | 1565 steps, loss 3.62 |
| SFT (Qwen) | `out/sft_qwen/final_model/` | 3750 steps |
| DPO (DeepSleep β=0.1) | `dpo_exp/ds_b0.1_s42/` | loss 0.02, acc 100% |
| DPO (DeepSleep β=0.5) | `dpo_exp/ds_b0.5_s42/` | loss 0.0001, acc 100% |
| DPO (Qwen β=0.1) | `dpo_exp/qwen_b0.1_s42/` | acc 100% |
| DPO (Qwen β=0.5) | `dpo_exp/qwen_b0.5_s42/` | acc 100% |

---

## 参考 & 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) — 轻量级 LLM 训练框架
- [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-moe/) — MoE 架构设计灵感
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — softmax routing + aux-loss
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 模型框架
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Benchmark 评估框架

---

## License

[MIT License](LICENSE)
