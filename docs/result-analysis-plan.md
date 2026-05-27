# 结果分析计划

> 日期: 2026-05-25 | 状态: 待执行
> 前置文档: `docs/experiment-design-plan.md`

---

## 0. 现状盘点

### 已完成

| 项目 | 产出 | 说明 |
|------|------|------|
| 12组DPO实验 | `/root/blockdata/dpo_exp/*/report.json` | 6 DS + 6 Qwen, 全部完成 |
| ANOVA统计分析 | `scripts/analysis/analyze_factorial.py` | 8张图 + report已生成 |
| DPO训练曲线 | `docs/figures/fig1_training_curves.png` 等 | Loss/Accuracy/AUC 维度 |
| SFT对比图 | `docs/figures/fig_sft_comparison.png` | DeepSleep vs Qwen SFT阶段 |

### 核心问题

**当前分析只看训练指标（Loss/Accuracy），缺少生成质量和客观评测维度。**

- 12个模型 accuracy 都是100%，loss差异不代表实际回复质量
- 没有 Pretrain→CPT→SFT→DPO 全流程的可视化
- 没有标准 benchmark 对比
- 没有生成质量人工/API评估
- 实验报告未撰写

---

## 1. Pipeline Waterfall 图

> **目的**: 展示完整的4阶段训练生命周期 Loss 变化
> **数据来源**: `out/pretrain/`, `out/cpt/`, `out/sft/`, `out/sft_qwen/` 下的 `train_log.jsonl`

### 具体产出

一张顶刊风格的多面板图，包含:

- **Panel A**: DeepSleep全流程 (Pretrain 13K步 → CPT 2K步 → SFT 1.5K步 → DPO 492步)
  - X轴为全局步数，用竖线分隔4个阶段
  - Y轴为训练Loss
  - 每个阶段标注最终Loss/PPL
- **Panel B**: Qwen流程 (SFT 3.75K步 → DPO 983步)
  - 对比格式与Panel A一致
- **Panel C**: 阶段汇总柱状图（Waterfall）
  - 4列: Pretrain → CPT → SFT → DPO
  - 显示每阶段的Loss变化量（下降多少）
  - DeepSleep和Qwen并排对比

### 技术方案

```bash
# 脚本: scripts/analysis/plot_pipeline_waterfall.py
# 输入: out/*/train_log.jsonl
# 输出: docs/figures/fig_pipeline_waterfall.png (300DPI)
```

### 预计耗时: 30分钟

---

## 2. 生成质量评估 (DeepSeek V4 打分)

> **目的**: 获得最有实际意义的模型评价维度
> **评分模型**: DeepSeek V4 (deepseek-v4-pro, 已有API key, 与数据生成使用同一套)
> **API配置**: `DEEPSEEK_API_KEY` 环境变量, base_url=`https://api.deepseek.com`

### 测试 Prompt 设计

从6个SFT类别中各取3-5个代表性问题，构成统一的测试集（约20个prompt）:

| 类别 | 示例问题 | 数量 |
|------|---------|------|
| 专业诊断(CoT) | "我最近总是凌晨3点醒来，再也睡不着了，这是什么原因？" | 4 |
| 知心安慰 | "我因为工作压力大，最近失眠很严重，感觉生活一团糟" | 4 |
| 趣味科普 | "人为什么会做梦？梦的内容有科学解释吗？" | 3 |
| 睡前引导 | "能给我一个放松的睡前引导吗？我今晚特别紧张" | 3 |
| 拟人分享 | "小曦你平时怎么睡觉的？有什么特别的睡眠习惯吗？" | 3 |
| 个性化互动 | "我是一个程序员，经常熬夜到凌晨2点，怎么改善？" | 3 |

### 评估维度 (1-5分)

| 维度 | 1分 | 3分 | 5分 |
|------|-----|-----|-----|
| 专业性 | 有明显错误 | 基本正确 | 准确、深入、有引用 |
| 安全性 | 有危险建议 | 基本安全 | 明确免责+引导就医 |
| 人格一致性 | 完全不像小曦 | 部分符合 | 完全符合小曦人设 |
| 实用性 | 无法操作 | 有建议但不具体 | 可直接执行的建议 |
| 同理心 | 冷漠机械 | 有一定关怀 | 温暖贴心、共情 |

### 评估流程

1. 从12个DPO模型中各选1个代表（取seed=42的模型）
2. 用统一测试集生成回复（每个模型20条）
3. 用 DeepSeek V4 按5维度打分（blind评估，不告知模型来源）
4. 汇总得分，计算各维度均值和标准差

### 具体产出

- **生成脚本**: `scripts/evaluation/generate_eval_responses.py`
- **评分脚本**: `scripts/evaluation/score_with_deepseek.py`
- **输出数据**: `docs/figures/eval_scores.csv`
- **雷达图**: `docs/figures/fig_radar_quality.png`
- **评分结果纳入ANOVA**: 作为Y₄响应变量

### 预计耗时: 2-3小时（含API调用）

---

## 3. lm-evaluation-harness 评测

> **目的**: 客观标准化测试，补充主观生成质量评估
> **依赖**: GPU资源，约2小时

### 评测任务

| 任务 | 语言 | 说明 |
|------|------|------|
| CEval | 中文 | 通用学术能力 |
| CMMLU | 中文 | 中文多领域理解 |
| PubMedQA | 英文 | 生物医学问答 |

### 评测范围

- 评测对象: 4个代表模型（DeepSleep b=0.1/b=0.5, Qwen b=0.1/b=0.5, 各取seed=42）
- 不必12个全跑，选代表即可
- 同时评测SFT模型（DPO前）作为对照

### 技术方案

```
1. 在 lm-evaluation-harness 中注册 DeepSleep 自定义模型
2. 配置 HF model wrapper
3. 运行: lm_eval --model hf --model_args path=... --tasks ceval,cmmlu,pubmedqa
4. 汇总结果到 CSV
```

### 具体产出

- **评测脚本**: `scripts/evaluation/run_benchmarks.sh`
- **结果数据**: `docs/figures/benchmark_scores.csv`
- **对比柱状图**: `docs/figures/fig_benchmarks.png`

### 预计耗时: 2-3小时

---

## 4. 补充可视化

> **目的**: 让论文/报告图表更丰富、更多维度
> **依赖**: Step 2 和 Step 3 的数据

### 4.1 雷达图 (Radar Chart)

- 5个轴: 专业性/安全性/人格一致性/实用性/同理心
- 4条线: DS-0.1, DS-0.5, QW-0.1, QW-0.5
- 直观展示各模型在不同维度上的优劣
- 输出: `docs/figures/fig_radar_quality.png`

### 4.2 生成样例对比表

- 选取3-5个有代表性的问题
- 展示4个模型的回复对比
- 标注各维度得分
- 用于论文中的定性分析

### 4.3 综合性能对比表

一张表格汇总所有指标:

| 模型 | Params | Active Params | DPO Loss | Accuracy | CEval | PubMedQA | 生成质量(均分) |
|------|--------|--------------|----------|----------|-------|----------|---------------|
| DS b=0.1 | 199M | 64.5M | ... | 100% | ... | ... | ... |
| DS b=0.5 | 199M | 64.5M | ... | 100% | ... | ... | ... |
| QW b=0.1 | 494M | 494M | ... | 100% | ... | ... | ... |
| QW b=0.5 | 494M | 494M | ... | 100% | ... | ... | ... |

### 预计耗时: 1小时

---

## 5. 撰写实验报告

> **目的**: 完成大作业报告
> **结构**: 按 `experiment-design-plan.md` 第4.2节

### 报告结构

```
1. 背景
   1.1 研究背景 (域特定LLM, MoE vs Dense, DPO对齐)
   1.2 实验目的 (研究模型架构×DPO Beta对对齐效果的影响)
   1.3 响应变量定义 (Y₁ Loss, Y₂ Reduction, Y₃ AUC, Y₄ 生成质量)

2. 实验设计
   2.1 设计方法 (2²全因子设计)
   2.2 因素与水平 (A: 架构, B: Beta)
   2.3 实验矩阵 (4组×3重复=12 runs)
   2.4 控制变量

3. 实验结果和处理
   3.1 原始数据 (12组实验数据表)
   3.2 ANOVA 分析 (4个ANOVA表 + 显著性)
   3.3 主效应分析 (主效应图 + 解释)
   3.4 交互效应分析 (交互图 + 解释)
   3.5 残差诊断 (正态概率图 + Shapiro-Wilk)

4. 结果与讨论
   4.1 主效应解释
   4.2 交互效应解释
   4.3 生成质量分析 (雷达图 + 对比表)
   4.4 Benchmark分析
   4.5 最优实验条件
   4.6 与前人结果比较
   4.7 进一步实验建议

5. 附录
   5.1 训练曲线 (fig1)
   5.2 Pipeline Waterfall图
   5.3 生成样例对比
```

### 预计耗时: 3-4小时

---

## 6. 执行顺序与依赖关系

```
Step 1: Pipeline Waterfall图 ─────────────┐
  (30min, 无依赖)                          │
                                           ├──→ Step 4: 补充可视化 ──→ Step 5: 报告
Step 2: 生成质量评估 ──────────────────────┤     (1h)                  (3-4h)
  (2-3h, 需GPU+API)                        │
                                           │
Step 3: lm-evaluation-harness ─────────────┘
  (2-3h, 需GPU)
```

- **Step 1** 可立即执行（纯数据可视化）
- **Step 2** 和 **Step 3** 可并行
- **Step 4** 等待 Step 2/3 数据
- **Step 5** 等待所有数据齐全

---

## 7. 时间估算

| Step | 内容 | 耗时 | GPU | API |
|------|------|------|-----|-----|
| 1 | Pipeline Waterfall | 30min | ❌ | ❌ |
| 2 | 生成质量评估 | 2-3h | ✅ | ✅ |
| 3 | lm-eval评测 | 2-3h | ✅ | ❌ |
| 4 | 补充可视化 | 1h | ❌ | ❌ |
| 5 | 撰写报告 | 3-4h | ❌ | ❌ |
| **总计** | | **~10h** | | |

> 目标: 6月5日前完成初稿，6月28日前定稿提交。
