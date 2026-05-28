# DeepSleep: Medical Sleep Health Domain LLM

## Project Overview

DeepSleep is a flexible MoE (Mixture-of-Experts) language model for the medical sleep health domain. Supports dense-only, all-MoE, and alternating architectures with optional shared experts. The persona "星辰曦（小曦）" is a warm, quirky sleep health companion.

**Architecture:** Qwen2.5-MoE / MiniMind inspired | **Framework:** HuggingFace | **Vocab:** 7200 BPE | **Context:** 8K tokens (extendable via RoPE)

> **重要原则：每完成一个任务，必须同步更新 CLAUDE.md 和 docs/ 下的计划文档。**

---

## Project Directory Structure

```
deepsleep/
├── model/
│   ├── __init__.py
│   └── model_deepsleep.py      # ALL model code: Config, Attention, MoE, CausalLM, Tokenizer
├── dataset/
│   ├── __init__.py
│   └── lm_dataset.py           # PretrainDataset, SFTDataset, DPODataset
├── trainer/
│   ├── __init__.py
│   ├── trainer_utils.py        # get_lr, Logger, init_model, lm_checkpoint, SkipBatchSampler
│   ├── train_pretrain.py       # ✅ 预训练 (HuggingFace Trainer, 流式CCI4.0-HQ, MoE-aware)
│   ├── train_sft.py            # ✅ SFT微调 (argparse + YAML config)
│   └── train_dpo.py            # ✅ DPO对齐 (argparse + YAML config)
├── configs/                         # 训练配置 (YAML)
│   ├── config_utils.py              # YAML config 加载器
│   ├── pretrain.yaml                # 预训练配置
│   ├── sft.yaml                     # SFT配置
│   └── dpo.yaml                     # DPO配置
├── scripts/
│   ├── run/                         # 一键启动脚本
│   │   ├── run_pretrain.sh          # 预训练
│   │   ├── run_sft.sh               # SFT微调
│   │   ├── run_dpo.sh               # DPO对齐
│   │   └── run_all.sh               # 全流程 (pretrain→SFT→DPO)
│   ├── generate_xiaoxi_all.py       # ✅ 小曦SFT数据生成 (两步分离: prompts → responses)
│   ├── generate_xiaoxi_dpo.py       # ✅ 小曦DPO对比数据 (两步分离: prompts → pairs, ChatML格式)
│   ├── prepare_sleep_corpus.py      # ✅ 睡眠语料筛选
│   ├── prepare_deepsleep_data.py    # ✅ 预训练数据下载 (从HuggingFace加载CCI4.0-HQ)
│   ├── train_tokenizer.py           # ✅ BPE分词器训练 (vocab=7200, 中英双语, 支持CCI4.0)
│   └── compare_tracks.py            # ✅ 双轨对比评估 (自包含, 含MCQ/CoT/人格/延迟)
├── server.py                        # ✅ 生产级 FastAPI 后端 (4模型切换, SSE流式, torch.compile FP8加速)
├── web/                             # ✅ React + Vite + Tailwind 前端 (DeepSeek风格UI)
│   ├── src/
│   │   ├── App.jsx                  # 根组件 (SSE流处理, 事件监听)
│   │   ├── components/              # UI 组件
│   │   │   ├── Header.jsx           # 顶栏: Logo + 模型架构/β下拉 + 设置
│   │   │   ├── Sidebar.jsx          # 侧边栏: 新对话 + 历史 + 睡眠评估/模型对比入口
│   │   │   ├── ChatArea.jsx         # 对话区域 + 自动滚动
│   │   │   ├── MessageBubble.jsx    # 消息气泡 + Markdown渲染 + Token统计
│   │   │   ├── ThinkingBlock.jsx    # 思考过程折叠面板 (脉冲动画)
│   │   │   ├── InputArea.jsx        # 输入框 + 思考开关 + 发送/停止
│   │   │   ├── SettingsPanel.jsx    # 参数设置: Temperature/Top-P/MaxTokens
│   │   │   ├── CompareMode.jsx      # 模型对比模式 (双栏并排)
│   │   │   ├── SleepAssessment.jsx  # 睡眠评估问卷 (简化版PSQI)
│   │   │   └── WelcomeScreen.jsx    # 欢迎页 + 快捷提问
│   │   ├── hooks/
│   │   │   ├── useChat.js           # Zustand: 对话CRUD + 设置 + localStorage
│   │   │   └── useModel.js          # 模型选择 (架构×β = 4种)
│   │   └── utils/
│   │       ├── api.js               # SSE流式请求 + fetchModels + fetchHealth
│   │       └── storage.js           # localStorage 封装
│   ├── vite.config.js               # outDir=../static, API proxy → :7860
│   └── package.json
├── static/                          # 前端构建产物 (npm run build → 此目录)
├── data/
│   ├── sft/xiaoxi/
│   │   ├── all_prompts.jsonl        # ✅ 10110条统一prompt (6类别)
│   │   ├── xiaoxi_sft.jsonl         # ✅ ChatML SFT数据 (10000条, 6类别各达目标)
│   │   └── .prompt_cache/           # 各类别prompt缓存
│   └── dpo/
│       ├── dpo_prompts.jsonl         # ✅ 2006条DPO prompt (6类别)
│       └── xiaoxi_dpo.jsonl          # ✅ 1965对 DPO对比数据
├── docs/                            # 设计文档, 计划
├── tests/                           # 单元测试
├── app.py                           # 旧版 Gradio web UI (保留)
├── Makefile
├── pyproject.toml
├── requirements.txt
└── .env                             # API credentials (gitignored)
```

---

## Key Commands

```bash
# Install
pip install -e ".[dev]"

# === 产品部署 (DeepSleep Chat) ===

# 启动生产级服务 (FastAPI + React前端)
python server.py --port 7860

# 启动时预加载所有模型 (推荐, 避免首请求延迟)
python server.py --port 7860 --preload

# 旧版 Gradio UI (保留)
python app.py --model /path/to/checkpoint

# 前端开发
cd web && npm run dev        # 开发模式 (热更新, 代理API到:7860)
cd web && npm run build      # 构建到 ../static/

# === Data Generation ===

python scripts/generate_xiaoxi_all.py --step prompts     # 生成SFT prompt
python scripts/generate_xiaoxi_all.py --step responses   # 生成SFT response
python scripts/generate_xiaoxi_all.py --step supplement  # 补充thinking样本
python scripts/generate_xiaoxi_all.py --step stats       # 查看统计

python scripts/generate_xiaoxi_dpo.py --step prompts     # 生成DPO prompt
python scripts/generate_xiaoxi_dpo.py --step pairs       # 生成DPO对比
python scripts/generate_xiaoxi_dpo.py --step stats       # 查看统计

python scripts/prepare_deepsleep_data.py                  # 预训练数据
python scripts/prepare_deepsleep_data.py --tokenizer_path checkpoints/tokenizer --validate

# === Training (single GPU, HuggingFace Trainer) ===

bash scripts/run/run_pretrain.sh                    # 预训练
bash scripts/run/run_sft.sh                         # SFT
bash scripts/run/run_dpo.sh                         # DPO
bash scripts/run/run_all.sh                         # 全流程

python trainer/train_pretrain.py --config configs/pretrain.yaml --tokenizer_path checkpoints/tokenizer
python trainer/train_sft.py --config configs/sft.yaml
python trainer/train_dpo.py --config configs/dpo.yaml

# Evaluation
python scripts/compare_tracks.py --track_a out/sft_a.pth --track_b out/sft_b.pth --tokenizer_path checkpoints/tokenizer

# Quick make targets
make pretrain ARGS="--data_path ... --tokenizer_path ..."
make sft ARGS="--data_path ... --from_weight ..."
make dpo ARGS="--data_path ... --sft_checkpoint ..."
```

---

## Model Architecture

### DeepSleep MoE (~200M params)

```
DeepSleepForCausalLM
├── Embedding (vocab=7200, d_model=768, tied with lm_head)
├── 8 MoE Layers (all MoE, no dense layers)
│   ├── DeepSleepAttention (GQA: 8Q/4KV heads, head_dim=96, RoPE, Flash/SDPA)
│   └── DeepSleepMoE (8 routed experts, top_k=2, intermediate=1216)
├── Final RMSNorm
└── LM Head (tied, no bias)

Total: ~199M params | Active per token: ~64.5M (32.4% utilization)
Special tokens: <thinking></thinking> for chain-of-thought reasoning

Mainstream components: GQA · RoPE · RMSNorm · SwiGLU · Flash Attention · Pre-Norm
```

### Flexible MoE Configuration

| Config | Description |
|--------|------------|
| `use_moe=False` | All dense layers |
| `use_moe=True, moe_layers=None` | All layers MoE (default) |
| `use_moe=True, moe_layers=[0,2,4..]` | Custom MoE layers |
| `num_shared_experts>0` | Always-active shared experts |

### Default Config

- d_model=768, n_layers=8, n_heads=8, n_kv_heads=4, head_dim=96
- MoE: 8 routed experts, 0 shared, top_k=2, intermediate=1216
- All 8 layers are MoE
- ~199M total params, ~64.5M active per token

### Legacy Checkpoint Compatibility

`DeepSleepConfig.from_legacy(config_dict)` maps old keys:
- `hidden_size` → `d_model`
- `num_hidden_layers` → `n_layers`
- `layer_pattern="all_moe"` → `moe_layers=[0..N]`
- `layer_pattern="alternating"` → `moe_layers=[odd indices]`

---

## Trained Model Artifacts

| Model | Path | Key Metrics |
|-------|------|-------------|
| Pretrain | `/root/autodl-tmp/data/deepsleep_model/final_model/` | 11,718 steps, loss 3.00 |
| SFT | `/root/autodl-tmp/data/deepsleep_model_sft/final_model/` | 32,625 steps, eval 1.84 |
| DPO | `/root/autodl-tmp/data/deepsleep_model_dpo_r2/final_model/` | 320 steps, acc 63.6% |

---

## 产品: DeepSleep Chat

### 产品概述

DeepSleep Chat 是一个 DeepSeek 风格的 AI 对话产品，接入训练完成的 4 个模型（DeepSleep/Qwen × β=0.1/0.5），支持公网部署。

**技术架构:**

```
浏览器 (React + Vite + Tailwind CSS)
    ↕ SSE (Server-Sent Events)
FastAPI 后端 (server.py)
    ↕ torch.compile (FP8 auto on RTX 4090 D / Ada Lovelace)
ModelManager (4个模型惰性加载 + 线程安全推理)
```

### 可用模型

| 模型 ID | 名称 | 架构 | HF路径 |
|---------|------|------|--------|
| `ds_b0.1` | DeepSleep β=0.1 | deepsleep | `out/ds_b0.1_hf/` |
| `ds_b0.5` | DeepSleep β=0.5 | deepsleep | `out/ds_b0.5_hf/` |
| `qwen_b0.1` | Qwen β=0.1 | qwen | `/root/blockdata/dpo_exp/qwen_b0.1_s42/final_model/` |
| `qwen_b0.5` | Qwen β=0.5 | qwen | `/root/blockdata/dpo_exp/qwen_b0.5_s42/final_model/` |

### 后端 (server.py)

**技术栈:** FastAPI + uvicorn + sse-starlette + torch + transformers

**核心特性:**
- **torch.compile FP8 加速**: 自动利用 RTX 4090 D (Ada Lovelace) FP8 tensor core，算子融合
- **惰性加载 + 模型预热**: 首次请求自动加载+预热（CUDA kernel编译），支持 `--preload` 启动时全加载
- **SSE 流式输出**: 逐字符推送，DeepSeek 风格打字机效果
- **思考模式**: 解析 `<thinking>...</thinking>` 标签，思考过程折叠展示
- **上下文截断**: 自动截断对话历史到模型最大上下文长度
- **线程安全**: 每个模型独立 asyncio.Lock，防止并发推理冲突
- **CORS**: 支持公网部署跨域访问
- **结构化日志**: Python logging 模块

**API 端点:**
- `POST /api/chat` — SSE 流式对话（请求体含 model/thinking/temperature/top_p/max_tokens）
- `GET /api/models` — 模型列表 + 加载状态
- `GET /api/health` — GPU 信息 + 显存 + 已加载模型

**推理优化:**
- DeepSleep: 手动逐 token 生成 + KV cache，每 2 token decode 一次（减少 tokenizer 开销）
- Qwen: `model.generate()` + `TextIteratorStreamer`（HF 原生流式，高度优化）
- 两者均通过 `torch.compile(mode="reduce-overhead")` 编译，自动 FP8

### 前端 (web/)

**技术栈:** React 18 + Vite 6 + Tailwind CSS 4 + Zustand

**UI 布局 (DeepSeek 风格深色主题):**
- 顶栏: Logo + 模型架构选择 (DeepSleep/Qwen) + β 值选择 (0.1/0.5) + 设置按钮
- 侧边栏: 新对话 + 对话历史（按日期分组）+ 删除对话 + 睡眠评估/模型对比入口
- 对话区: 用户紫色气泡 / AI 深灰气泡 + Markdown 渲染 + 思考过程折叠面板 + Token 统计
- 输入区: 自动增长 textarea + Enter 发送 + 思考模式开关 + 发送/停止按钮
- 欢迎页: 小曦介绍 + 6 个快捷提问卡片
- 设置面板: Temperature / Top-P / 最大生成长度滑块

**创新功能:**
1. **模型对比模式 (CompareMode)** — 选择两个模型，同一问题双栏并排对比回答
2. **睡眠评估工具 (SleepAssessment)** — 7 题简化版 PSQI 问卷，自动评分，一键让 AI 分析
3. **深色/浅色主题切换**
4. **对话导出** — 导出 Markdown 文件
5. **对话持久化** — localStorage 自动保存

**构建:** `npm run build` → 输出到 `../static/`，由 FastAPI 托管静态文件

---

## Data Pipeline

```
CCI4.0-HQ (HuggingFace) → train_pretrain.py 流式加载 (无需下载到本地)

DeepSleep Pipeline:
  Pretrain: CCI4.0-HQ 流式 → DeepSleep MoE (~199M params)
  SFT:      generate_xiaoxi_all.py → ✅ 6类 10000条 ChatML
  DPO:      generate_xiaoxi_dpo.py → ✅ 1965对 6类别+6反面风格 ChatML

  小曦人格SFT 6类数据 (10000条已达标):
    1. 专业诊断CoT  2500条  (含 <thinking></thinking> 推理)
    2. 知心安慰     2500条
    3. 趣味科普     1500条
    4. 睡前引导     1000条
    5. 拟人分享     1000条
    6. 个性化互动   1500条
```

---

## Data Generation Status

### BPE 分词器 (train_tokenizer.py)

| 属性 | 值 |
|------|-----|
| 类型 | BPE (Byte-Pair Encoding) |
| Vocab size | 7200 |
| 特殊 tokens | `<pad>(0)`, `<\|im_start\|>(1)`, `<\|im_end\|>(2)`, `<unk>(3)`, `<s>(4)`, `</s>(5)`, `<thinking>(6)`, `</thinking>(7)`, `<summary>(8)` |
| Normalizer | 无 (byte-level BPE) |
| Pre-tokenizer | GPT-4 style regex split |
| ChatML 模板 | `<\|im_start\|>role\ncontent<\|im_end\|>\n` |

**训练语料：**

| 数据源 | 语言 | 内容 | 条数 |
|--------|------|------|------|
| `data/sft/xiaoxi/xiaoxi_sft.jsonl` | 中文 | 小曦SFT对话 (6类) | ~10000 |
| `data/dpo/xiaoxi_dpo.jsonl` | 中文 | 小曦DPO偏好对比 | ~1965 |
| `data/sft/xiaoxi/all_prompts.jsonl` | 中文 | SFT prompt (6类) | ~10110 |
| `pubmedqa/data/ori_pqal.json` | 英文 | PubMed生物医学问答 | 1000 |
| **合计** | | | **~20081 segments** |

> **说明**: 由于网络不可用，未下载额外英文语料。英文覆盖主要靠 PubMedQA。后续可补充英文语料重新训练。

```bash
python scripts/train_tokenizer.py  # 重新训练分词器
```

### SFT 数据生成 (generate_xiaoxi_all.py)

| 步骤 | 状态 | 详情 |
|------|------|------|
| Prompt生成 (6类别) | ✅ 完成 | 10110条已缓存 |
| Response生成 | ✅ 完成 | 10000条已生成, 6类别各达目标 |

> **2026-05-22 完成**: SFT数据扩容至10000条全部完成。各类别均达标：专业诊断2500、知心安慰2500、趣味科普1500、睡前引导1000、拟人分享1000、个性化互动1500。

| 类别 | 条数 | 目标 | 状态 |
|------|------|------|------|
| 专业诊断(CoT) | 2500 | 2500 | ✅ |
| 知心安慰 | 2500 | 2500 | ✅ |
| 趣味科普 | 1500 | 1500 | ✅ |
| 睡前引导 | 1000 | 1000 | ✅ |
| 拟人分享 | 1000 | 1000 | ✅ |
| 个性化互动 | 1500 | 1500 | ✅ |
| **合计** | **10000** | **10000** | **✅** |

### DPO 数据生成 (generate_xiaoxi_dpo.py)

| 步骤 | 状态 | 详情 |
|------|------|------|
| Prompt生成 (6类别) | ✅ 完成 | 2006条 |
| Pair生成 | ✅ 完成 | 1965/2006 对 (chosen+rejected) |

| 类别 | 条数 |
|------|------|
| 专业诊断(CoT) | 800 |
| 知心安慰 | 403 |
| 趣味科普 | 300 |
| 睡前引导 | 163 |
| 拟人分享 | 150 |
| 个性化互动 | 149 |
| **合计** | **1965** |

输出格式：`messages`(共享前缀) + `chosen`/`rejected`(string)，直接对接 `DPODataset`。

### 预训练数据 (流式加载 CCI4.0-HQ)

| 步骤 | 状态 | 详情 |
|------|------|------|
| CCI4.0-HQ 流式预训练 | ❌ 未开始 | train_pretrain.py 直接从 HF 流式加载，无需下载 |

> 预训练采用 HuggingFace 流式加载（`CCI4PretrainDataset`），无需本地下载。也可用 `scripts/prepare_deepsleep_data.py` 提前下载到本地。

---

## Training Pipeline

```
Stage 1: Pretrain ✅ → out/pretrain/ (13K steps, loss 1.77, PPL 5.85)
Stage 2: CPT ✅ → out/cpt/ (2K steps, loss 1.14, PPL 3.11)
Stage 3a: SFT (DeepSleep) ✅ → out/sft/ (1565 steps, loss 3.62)
Stage 3b: SFT (Qwen) ✅ → out/sft_qwen/ (3750 steps)
Stage 4a: DPO (DeepSleep) ✅ → /root/blockdata/dpo_exp/ds_* (6组完成)
Stage 4b: DPO (Qwen) ✅ → /root/blockdata/dpo_exp/qwen_* (6组完成)
Stage 5: lm-eval Benchmark ✅ → data/eval/benchmark_results/ (8模型×5 benchmark)

《科学实验分析》大作业: 2²全因子设计 × 3重复 = 12组DPO实验
  详细计划: docs/experiment-design-plan.md
  实验脚本: scripts/run/exp/*.sh
  结果目录: /root/blockdata/dpo_exp/

---

## Known Issues

### `train_sft.py` 缺 LoRA 支持

计划添加 `--use_lora` 参数，暂未实现。

---

## Next Steps (Remaining Work)

### DeepSleep Chat 产品 — 已完成 ✅

产品级 AI 对话界面已完成开发，包含:
- ✅ 生产级 FastAPI 后端 (server.py)
- ✅ React + Vite + Tailwind 前端 (web/)
- ✅ 4 模型切换 (DeepSleep/Qwen × β=0.1/0.5)
- ✅ 思考/非思考模式
- ✅ SSE 流式逐字输出
- ✅ torch.compile FP8 加速
- ✅ 模型对比模式
- ✅ 睡眠评估工具
- ✅ 对话历史持久化
- ✅ 深色/浅色主题
- ✅ 公网部署 (CORS)

### 《科学实验分析》大作业 — 2²全因子DPO实验

> 详细计划见: `docs/experiment-design-plan.md`

1. ~~**Pretrain + CPT + SFT (DeepSleep)**~~ — ✅ 全部完成
2. ~~**SFT (Qwen2.5-0.5B)**~~ — ✅ 完成, `out/sft_qwen/final_model`
3. ~~**DPO脚本修复+Qwen DPO创建**~~ — ✅ 完成
4. ~~**DeepSleep DPO 6组实验**~~ — ✅ 完成, beta=0.1 loss~0.04, beta=0.5 loss~0.0006
5. ~~**Qwen DPO 6组实验**~~ — ✅ 完成
6. ~~**收集12组 report.json**~~ — ✅ 完成, `/root/blockdata/dpo_exp/*/report.json`
7. ~~**lm-evaluation-harness 评估**~~ — ✅ 完成, 8模型 × 5 benchmark, `docs/figures/benchmark_results.csv`
8. ~~**2²全因子 ANOVA 统计分析**~~ — ✅ 完成, `scripts/analysis/analyze_factorial.py`
   - Fig 0-8 共 9 张图 (训练曲线/主效应/交互/残差/Box/Pareto/Heatmap/收敛)
   - Markdown 分析报告: `docs/analysis_report.md`
9. ~~**Benchmark acc_norm 多因素 ANOVA**~~ — ✅ 完成, `scripts/analysis/plot_benchmark_anova.py`
   - 无重复双因素: Model(8) × Benchmark(5), 手动计算+statsmodels验证
   - 分组柱状图: `fig_benchmark_anova_barplot.png/pdf`
10. ~~**生成质量评估**~~ — ✅ 完成, `scripts/evaluation/eval_quality.py`
    - 30条prompt × 4模型, DeepSeek V4 按10维度打分
    - 柱状图: `fig_quality_anova.png`
11. ~~**可视化 (顶刊风格)**~~ — ✅ 完成, 雷达图/ANOVA图/Waterfall图/对比表, 300DPI
12. ⬜ **撰写实验报告** — 按 `docs/experiment-design-plan.md` 第4.2节结构

> 作业截止: **6月28日** | 选题讨论: **6月3日/10日** | 目标6月5日前完成初稿

### 实验关键文件

| 文件 | 说明 |
|------|------|
| `trainer/train_dpo.py` | DeepSleep DPO (含JSONL日志+accuracy+report) |
| `trainer/train_dpo_qwen.py` | Qwen DPO (含del logits内存优化) |
| `scripts/run/exp/*.sh` | 12个独立实验脚本 (ds_b0.1_s42.sh 等) |
| `/root/blockdata/dpo_exp/` | 12组DPO实验结果 (report.json + logs) |
| `docs/experiment-design-plan.md` | 作业实验计划文档 |
| `scripts/eval/convert_to_hf.py` | DeepSleep .pth → HF格式转换 |
| `scripts/eval/run_benchmark.sh` | lm-eval 8模型×5 benchmark 评估脚本 |
| `scripts/eval/merge_results.py` | 合并增量benchmark结果到已有文件 |
| `scripts/eval/plot_radar.py` | 雷达图+CSV数据表生成 |
| `scripts/analysis/analyze_factorial.py` | 2²全因子ANOVA (训练loss/AUC等响应) |
| `scripts/analysis/plot_benchmark_anova.py` | Benchmark acc_norm 双因素ANOVA+柱状图 |
| `scripts/evaluation/eval_quality.py` | 生成质量评估 (DeepSeek V4 10维度打分) |
| `data/eval/benchmark_results/` | lm-eval 评估结果目录 |

---

## Update Log

- **2026-05-28**: 产品封顶 — 完成 DeepSleep Chat 对话产品。后端 `server.py` (FastAPI + SSE 流式 + torch.compile FP8 加速 + 模型预热 + CORS + 结构化日志)。前端 React + Vite + Tailwind (DeepSeek 风格深色主题, 10个组件: Header/Sidebar/ChatArea/MessageBubble/ThinkingBlock/InputArea/SettingsPanel/CompareMode/SleepAssessment/WelcomeScreen, Zustand 状态管理, localStorage 持久化)。支持 4 模型切换 (DeepSleep/Qwen × β=0.1/0.5), 思考/非思考模式, Temperature/Top-P/MaxTokens 调节, 模型对比, 睡眠评估问卷, 深色/浅色主题, 对话导出。3 Agent 并行开发 + 1 Agent 审查集成。
- **2026-05-27**: 大作业基本完成 — Phase 3-5 全部完成。2²全因子ANOVA (`analyze_factorial.py`, 9张图+报告)。Benchmark acc_norm 双因素ANOVA (`plot_benchmark_anova.py`, 无重复设计Model×Benchmark, 手动计算+statsmodels验证+Tukey HSD+分组柱状图)。生成质量评估 (`eval_quality.py`, 30prompt×4模型×10维度, DeepSeek V4打分, ANOVA柱状图)。雷达图+Pipeline Waterfall图+风格统一(去除加粗等)。仅剩撰写实验报告。
- **2026-05-27**: Phase 4 lm-evaluation-harness 评估进行中 — 8模型(DS Pretrain, DS β=0.1/0.5, MiniMind-3, Medical-GPT2, Qwen Base, Qwen β=0.1/0.5) × 5 benchmark(PubMedQA本地, MedQA, ARC-Easy, PIQA, OpenBookQA)。创建 `scripts/eval/` 评估工具集(convert_to_hf.py, run_benchmark.sh, merge_results.py, plot_radar.py)。DeepSleep .pth 转 HF 格式+auto_map注册。PubMedQA 自定义 lm-eval 任务(pubmedqa_local)绕过 datasets 4.x 脚本限制。4个DPO模型已有结果(PubMedQA/MedQA/ARC-Easy/PIQA)，正在补充 OpenBookQA + 新增4个基线模型。
- **2026-05-25**: 《科学实验分析》大作业实验执行 — 2²全因子设计(模型架构×DPO Beta, 3重复=12 runs)。修复 train_dpo.py 添加 JSONL 日志/accuracy/report.json。创建 train_dpo_qwen.py 和 dpo_qwen.yaml。修复 Qwen DPO OOM (del logits 释放中间张量 + batch_size=2)。DeepSleep DPO 6组全部完成(beta=0.1 loss~0.04, beta=0.5 loss~0.0006, accuracy均100%)。Qwen DPO 6组完成。详细计划见 `docs/experiment-design-plan.md`。
- **2026-05-25**: Qwen SFT 完成 — 3750 steps, loss 3.82→~1.5, 模型保存在 `out/sft_qwen/final_model` (HF格式, ~1.9GB)。
- **2026-05-24**: Pretrain+CPT+SFT(DeepSleep) 全部完成 — Pretrain 13K steps loss 1.77/PPL 5.85, CPT 2K steps loss 1.14/PPL 3.11, SFT 1565 steps loss 3.62。
- **2026-05-22**: 架构确定为199M/64.5M MoE + 工业级训练配置 — 模型改为8层全MoE(8 routed experts, 0 shared, top_k=2, intermediate=1216) → ~199M total / ~64.5M active。创建 `configs/` 目录(pretrain/sft/dpo YAML配置 + config_utils.py加载器)。创建 `scripts/run/` 一键启动脚本(run_pretrain/sft/dpo/all.sh)。更新 README.md 完整重写。添加 MIT LICENSE。更新 .gitignore, Makefile, requirements.txt。
- **2026-05-22**: 预训练脚本重写 + 模型架构修复 — 用 HuggingFace Trainer 重写 `train_pretrain.py`（流式CCI4.0-HQ, MoE-aware loss, checkpoint/resume, TensorBoard, 样本生成回调）。新建 `dataset/streaming_dataset.py` 流式数据集。修改 `scripts/train_tokenizer.py` 支持 CCI4.0 中英文语料。修复模型 intermediate_size 默认值（dense=2048, MoE=1472），总参数从 104M 修正为 ~182M。统一所有训练脚本 vocab_size=7200。`train_tokenizer.py` 从 trainer/ 移到 scripts/。
- **2026-05-22**: SFT数据10000条全部完成 + DPO数据1965对 — SFT 6类别各达目标(专业诊断2500、知心安慰2500、趣味科普1500、睡前引导1000、拟人分享1000、个性化互动1500)。DPO数据扩容至1965对(专业800、知心403、趣味300、睡前163、拟人150、互动149)。DPO prompt 2006条。全部数据生成完成，进入训练准备阶段。下一任务：预训练数据准备 + Track B Pretrain。
- **2026-05-22**: Tokenizer训练 + 预训练测试 — 训练 7200 vocab BPE 分词器 (语料: SFT/DPO/PubMedQA ~15969 segments)。创建 `train_tokenizer.py` 和 `test_pretrain.py` 脚本。预训练流程验证通过 (123M params, 75 steps, loss 10.48→0.38)。模型默认 vocab_size 改为 7200。
- **2026-05-22**: SFT数据扩容至10000条 — 将 `ALL_CATEGORIES` 目标从 3500 提升至 10000（专业诊断2500、知心安慰2500、趣味科普1500、睡前引导1000、拟人分享1000、个性化互动1500）。将 prompt 生成改为全并发模式：拆分 `_generate_one_batch` 单批生成函数，`generate_all_prompts` 将所有批次一次性提交到 ThreadPoolExecutor（默认20 workers），与 response 生成一致。断点续生：已有 prompt 去重追加，已有 response 跳过。
- **2026-05-22**: 修复SFT thinking不一致 — 专业诊断(CoT) 99.4% 含 thinking，其他5类无 thinking（与 DPO 数据格式不一致）。修改 `generate_xiaoxi_all.py`：所有类别 system_prompt 加入 thinking 格式要求，`_generate_one` 增加 thinking 合规校验+重试，新增 `--step supplement` 模式为非CoT类别补充 ~310 条带 thinking 数据。更新 CLAUDE.md。
- **2026-05-22**: 修复脚本问题 — 删除废弃的 `generate_cot_data.py` 和 `generate_dpo_data.py`（已被 `generate_xiaoxi_all.py` 和 `generate_xiaoxi_dpo.py` 替代）；重写 `compare_tracks.py` 为自包含脚本（嵌入MCQ题目、评估逻辑，使用 DeepSleep 模型加载）；重写 `prepare_deepsleep_data.py` 改为从 HuggingFace 加载 CCI4.0-HQ 子集并采样 ~12B tokens；更新 CLAUDE.md 和计划文档。
- **2026-05-21**: DPO脚本重写 — 修复格式(输出ChatML消息列表), 复用XIAOXI_IDENTITY完整人格, 覆盖6类别(含思考链), 6种rejected风格, 默认1500条, 并发生成。
- **2026-05-21**: SFT数据生成完成 (3564条responses全部生成), DPO/预训练数据未开始。
- **2026-05-21**: SFT数据生成中 (448/3500 responses), DPO/预训练数据未开始。更新CLAUDE.md和计划文档。
- **2026-05-21**: 架构重构 — 合并8个模型文件到 `model_deepsleep.py`, 废弃YAML/OmegaConf改用argparse, 修复所有review bug, 采用MiniMind模式。创建 `generate_xiaoxi_all.py` (两步分离SFT生成) 和 `generate_xiaoxi_dpo.py` (DPO对比)。创建 `prepare_deepsleep_data.py` (预训练数据下载)。`.env` 配置API key。
- **2026-05-20**: 创建MoE多共享专家, think tokens, CoT/DPO生成器, 评估基准, 小曦人格定义
- **2026-04-27**: DPO训练完成 (320 steps, 63.6% accuracy)
- **2026-04-26**: SFT训练完成 (32,625 steps, eval loss 1.84)
- **2026-04-25**: Pretrain完成 (11,718 steps, loss 3.00)
- **2026-04-07**: 项目初始化
