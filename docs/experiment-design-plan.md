# DeepSleep vs Qwen: 域特定LLM训练的2²全因子实验设计

> 《科学实验分析》大作业实验计划
> 日期: 2026-05-25 | 更新: 2026-05-25

---

## 1. 背景

### 1.1 研究背景

大语言模型 (LLM) 在通用领域取得了巨大成功，但在垂直领域（如医疗健康）的应用仍面临挑战。域特定LLM需要在有限的领域数据上实现高质量的专业知识生成，同时保持模型的可控性和安全性。

本实验以**睡眠健康**为垂直领域，研究两个核心问题：

1. **模型架构的影响**：稀疏混合专家 (MoE) 与稠密 (Dense) 架构在域特定对齐任务上的差异
2. **对齐强度的影响**：DPO (Direct Preference Optimization) 中 Beta 参数对偏好对齐效果的影响

### 1.2 实验目的

通过系统的 2² 全因子实验设计，研究模型架构和DPO对齐强度两个因素及其交互作用对模型对齐效果的影响，找出最优的模型-超参数组合。

### 1.3 实验响应（评价指标）

| 响应变量 | 说明 | 采集方式 |
|----------|------|----------|
| **Y₁: DPO Loss** | DPO训练最终收敛损失 | 训练日志自动记录 |
| **Y₂: 对齐准确率** | chosen logps > rejected logps 的比例 | 推理计算 |
| **Y₃: 生成质量评分** | 模型生成回复的质量 (1-5分) | GPT-4o自动评估 |

---

## 2. 实验设计

### 2.1 实验设计方法

采用 **2² 全因子设计 (Full Factorial Design)**，2个因素各2个水平，共4种处理组合。

选择全因子设计的原因：
- 可以估计所有主效应和交互效应
- 实验量小（4组处理），适合有限GPU资源
- ANOVA分析方法成熟，结论可靠

### 2.2 实验因素与水平

| 因素 | 名称 | 水平1 (-1) | 水平2 (+1) |
|------|------|------------|------------|
| **A** | 模型架构 | DeepSleep MoE (199M, 64.5M active) | Qwen2.5-0.5B Dense (494M) |
| **B** | DPO Beta | 0.1 (弱对齐) | 0.5 (强对齐) |

**因素选择理由：**
- **因素A (模型架构)**: MoE通过稀疏激活实现参数效率，Dense是主流架构。对比两者在域特定场景下的对齐能力是有价值的研究问题。
- **因素B (DPO Beta)**: Beta控制偏好对齐的强度。Beta=0.1是常用默认值（温和对齐），Beta=0.5是更强的对齐信号，可能导致过度对齐或提升效果。

### 2.3 实验矩阵

| Group | 模型架构 | DPO Beta | 编码 |
|-------|---------|----------|------|
| 1 | DeepSleep MoE | 0.1 | (-1, -1) |
| 2 | DeepSleep MoE | 0.5 | (-1, +1) |
| 3 | Qwen Dense | 0.1 | (+1, -1) |
| 4 | Qwen Dense | 0.5 | (+1, +1) |

### 2.4 重复实验

每组处理重复 **n=3** 次（随机种子：42, 123, 7），共 **12** 次实验。

重复的目的：
- 估计实验误差（纯误差）
- 增加统计检验的功效（误差自由度=8）
- 使 ANOVA 的 F 检验更可靠

### 2.5 完整实验计划表

| Run | 模型 | Beta | Seed | SFT起点 | DPO输出目录 | 启动脚本 |
|-----|------|------|------|---------|-------------|----------|
| 1 | DeepSleep | 0.1 | 42 | out/sft/final_model.pth | /root/blockdata/dpo_exp/ds_b0.1_s42 | scripts/run/exp/ds_b0.1_s42.sh |
| 2 | DeepSleep | 0.1 | 123 | out/sft/final_model.pth | /root/blockdata/dpo_exp/ds_b0.1_s123 | scripts/run/exp/ds_b0.1_s123.sh |
| 3 | DeepSleep | 0.1 | 7 | out/sft/final_model.pth | /root/blockdata/dpo_exp/ds_b0.1_s7 | scripts/run/exp/ds_b0.1_s7.sh |
| 4 | DeepSleep | 0.5 | 42 | out/sft/final_model.pth | /root/blockdata/dpo_exp/ds_b0.5_s42 | scripts/run/exp/ds_b0.5_s42.sh |
| 5 | DeepSleep | 0.5 | 123 | out/sft/final_model.pth | /root/blockdata/dpo_exp/ds_b0.5_s123 | scripts/run/exp/ds_b0.5_s123.sh |
| 6 | DeepSleep | 0.5 | 7 | out/sft/final_model.pth | /root/blockdata/dpo_exp/ds_b0.5_s7 | scripts/run/exp/ds_b0.5_s7.sh |
| 7 | Qwen | 0.1 | 42 | out/sft_qwen/final_model | /root/blockdata/dpo_exp/qwen_b0.1_s42 | scripts/run/exp/qwen_b0.1_s42.sh |
| 8 | Qwen | 0.1 | 123 | out/sft_qwen/final_model | /root/blockdata/dpo_exp/qwen_b0.1_s123 | scripts/run/exp/qwen_b0.1_s123.sh |
| 9 | Qwen | 0.1 | 7 | out/sft_qwen/final_model | /root/blockdata/dpo_exp/qwen_b0.1_s7 | scripts/run/exp/qwen_b0.1_s7.sh |
| 10 | Qwen | 0.5 | 42 | out/sft_qwen/final_model | /root/blockdata/dpo_exp/qwen_b0.5_s42 | scripts/run/exp/qwen_b0.5_s42.sh |
| 11 | Qwen | 0.5 | 123 | out/sft_qwen/final_model | /root/blockdata/dpo_exp/qwen_b0.5_s123 | scripts/run/exp/qwen_b0.5_s123.sh |
| 12 | Qwen | 0.5 | 7 | out/sft_qwen/final_model | /root/blockdata/dpo_exp/qwen_b0.5_s7 | scripts/run/exp/qwen_b0.5_s7.sh |

### 2.6 控制变量（固定不变）

| 变量 | DeepSleep | Qwen | 说明 |
|------|-----------|------|------|
| DPO数据 | xiaoxi_dpo.jsonl (1965对) | 同左 | 同一份数据 |
| DPO epochs | 1 | 1 | DPO标准做法 |
| batch_size | 4 | 2 | Qwen显存限制 |
| accumulation | 4 | 8 | 有效batch均为16 |
| LR | 5e-7 | 5e-7 | 一致 |
| max_seq_len | 3072 | 3072 | 一致 |
| 数据shuffle种子 | 按Run | 按Run | 保证重复性 |

### 2.7 已完成的 DeepSleep DPO 结果

| Beta | Seed | Final Loss | Accuracy | 时间 |
|------|------|-----------|----------|------|
| 0.1 | 42 | 0.0207 | 100% | 9min |
| 0.1 | 123 | 0.0614 | 100% | 12min |
| 0.1 | 7 | 0.0404 | 100% | 9min |
| **0.1 均值** | | **0.0408** | **100%** | |
| 0.5 | 42 | 0.0001 | 100% | 11min |
| 0.5 | 123 | 0.0012 | 100% | 8min |
| 0.5 | 7 | 0.0006 | 100% | 8min |
| **0.5 均值** | | **0.0006** | **100%** | |

**观察**: Beta=0.5 loss 远低于 0.1，但这是因为 beta 放大 logits_diff 导致 sigmoid 更快饱和，不代表生成质量更好。两个 beta 的 accuracy 均为 100%。

---

## 3. 实验结果和处理

### 3.1 数据采集

每次DPO训练记录以下指标：

**训练日志 (train_log.jsonl):**
```json
{"step": 50, "loss": 0.4974, "accuracy": 1.0, "lr": 5.0e-7}
```

**评估日志 (eval_log.jsonl):**
```json
{"step": 492, "eval_loss": 0.065, "eval_accuracy": 0.98}
```

**最终报告 (report.json):**
```json
{
  "model": "DeepSleep-MoE",
  "dpo_beta": 0.1,
  "seed": 42,
  "final_loss": 0.0207,
  "final_accuracy": 1.0,
  "total_steps": 492,
  "total_time_hours": 0.15
}
```

### 3.2 统计分析方法

#### (1) ANOVA 方差分析

对每个响应变量 (Y₁, Y₂, Y₃) 分别进行 2² 全因子 ANOVA：

**线性统计模型：**
```
Y = μ + α·A + β·B + (αβ)·A×B + ε
```

其中：
- μ: 总均值
- α: 因素A (模型架构) 的主效应
- β: 因素B (DPO Beta) 的主效应
- (αβ): A×B 交互效应
- ε: 随机误差

**ANOVA 表结构：**

| 来源 | 自由度 df | 平方和 SS | 均方 MS | F值 | p值 |
|------|----------|----------|---------|-----|-----|
| A (模型架构) | 1 | SS_A | MS_A | F_A | p_A |
| B (DPO Beta) | 1 | SS_B | MS_B | F_B | p_B |
| A×B (交互) | 1 | SS_AB | MS_AB | F_AB | p_AB |
| 误差 | 8 | SS_E | MS_E | | |
| 总计 | 11 | SS_T | | | |

显著性水平 α = 0.05。

#### (2) 主效应分析

计算各因素在每个水平下的平均响应值，绘制主效应图 (Main Effects Plot)。

#### (3) 交互效应分析

绘制交互效应图 (Interaction Plot):
- 横轴: 因素B的水平 (Beta)
- 纵轴: 响应变量的均值
- 两条线: 分别代表 DeepSleep 和 Qwen
- 若两线不平行 → 存在交互效应

#### (4) 残差诊断

- 残差的正态概率图 (Normal Probability Plot)
- 残差 vs 拟合值图
- 验证 ANOVA 假设（正态性、等方差性、独立性）

### 3.3 生成质量评估

对每组实验的最终模型，使用统一的测试 prompt 集进行生成，由 GPT-4o 按以下维度打分 (1-5分)：

| 维度 | 说明 |
|------|------|
| 专业性 | 回复是否准确、专业 |
| 安全性 | 是否有不当医疗建议 |
| 人格一致性 | 是否符合"小曦"人设 |
| 实用性 | 是否给出可操作的建议 |
| 同理心 | 是否体现关怀 |

### 3.4 Benchmark acc_norm 多因素方差分析

#### 分析动机

DPO 训练的 4 个模型 (DeepSleep β=0.1/0.5, Qwen β=0.1/0.5) 加上 4 个基线模型 (DeepSleep-Base, MiniMind-3, Medical-GPT2, Qwen2.5-Base)，在 5 个标准 benchmark 上获得了 acc_norm 分数。为了回答"**模型之间的性能差异是否显著**"这个问题，我们进行了多因素方差分析。

#### 实验设计

这是一个**无重复双因素方差分析** (Two-Way ANOVA without replication)，等价于**随机化完全区组设计** (Randomized Complete Block Design)。

| 项目 | 说明 |
|------|------|
| 因子 A (Model) | 8 个水平: 4 个 DPO 模型 + 4 个基线 |
| 因子 B (Benchmark) | 5 个水平: PubMedQA, MedQA, ARC-Easy, PIQA, OpenBookQA |
| 响应变量 | acc_norm 分数 (%) |
| 总观测数 | 8 × 5 = 40 |
| 每格重复数 | 1 (无重复) |

#### 核心限制与处理

由于每个 (模型, benchmark) 组合只有 1 个分数（没有重复实验），我们**无法将交互效应与随机误差分开**。因此：

- **交互效应 = 误差项**：SS_Error = SS_Total - SS_Model - SS_Benchmark，实际上就是 Model × Benchmark 交互作用的平方和
- **F 检验的零假设**：
  - H₀(Model): 所有模型的平均 acc_norm 相同（模型选择对性能无影响）
  - H₀(Benchmark): 所有 benchmark 的平均 acc_norm 相同（benchmark 难度无差异）
- **F 检验的分母**：MS_Error = MS_Interaction，这意味着我们实际上是将因子效应与交互效应做比较

#### 计算公式

```
SS_Total = Σᵢⱼ (yᵢⱼ - ȳ··)²                                    df = ab-1 = 39
SS_Model = b × Σᵢ (ȳᵢ· - ȳ··)²           (b=5, 每个模型在5个benchmark上的均值)   df = a-1 = 7
SS_Bench = a × Σⱼ (ȳ·ⱼ - ȳ··)²           (a=8, 每个benchmark在8个模型上的均值)   df = b-1 = 4
SS_Error = SS_Total - SS_Model - SS_Bench  (= Model×Benchmark 交互)              df = (a-1)(b-1) = 28

F_Model = MS_Model / MS_Error = (SS_Model/7) / (SS_Error/28)     ~ F(7, 28)
F_Bench = MS_Bench / MS_Error = (SS_Bench/4) / (SS_Error/28)     ~ F(4, 28)
```

#### 效应量

- **η² (eta-squared)** = SS_factor / SS_Total，表示该因子解释了总变异的百分比
- 例如 η²(Model)=0.7 意味着 70% 的分数变异是由模型不同造成的

#### 输出图表

- **分组柱状图** (`fig_benchmark_anova_barplot.png/pdf`): X轴=5个benchmark, 每组8根柱子代表8个模型, 按家族分色(DeepSleep蓝/Qwen青蓝/Baseline暖色), 每组最高分柱子用黑色细线框标注
- **Tukey HSD 事后检验**: 模型间配对比较，找出哪些模型对之间存在显著差异

#### 脚本

```bash
python scripts/analysis/plot_benchmark_anova.py
```

---

## 4. 结果与讨论

### 4.1 预期分析内容

1. **主效应结论**: 模型架构和DPO Beta哪个因素对对齐效果影响更大？
2. **交互效应**: 最优Beta值是否取决于模型架构？
3. **最优组合**: 哪种模型+Beta组合的对齐效果最好？
4. **与前人比较**: 与 DPO 论文 (Rafailov et al., 2023) 和 MoE 相关工作对比
5. **进一步实验建议**: 若交互效应显著，是否需要响应曲面设计找最优点？

### 4.2 论文/报告结构

```
1. 背景
   1.1 研究背景
   1.2 实验目的
   1.3 响应变量定义

2. 实验设计
   2.1 设计方法选择
   2.2 因素与水平
   2.3 实验矩阵
   2.4 控制变量

3. 实验结果和处理
   3.1 原始数据
   3.2 ANOVA 分析
   3.3 主效应分析
   3.4 交互效应分析
   3.5 残差诊断

4. 结果与讨论
   4.1 主效应解释
   4.2 交互效应解释
   4.3 最优实验条件
   4.4 与前人结果比较
   4.5 进一步实验建议

5. 附录
   5.1 训练曲线
   5.2 生成样例
```

---

## 5. 实施步骤

### ~~Phase 1: 基础设施准备~~ ✅ 已完成

1. ✅ **修复 DPO 脚本**: `train_dpo.py` 已添加 JSONL 日志、accuracy、report.json
2. ✅ **创建 Qwen DPO 脚本**: `train_dpo_qwen.py` 已创建
3. ✅ **创建独立实验脚本**: `scripts/run/exp/` 下 12 个独立脚本
4. ✅ **Qwen SFT 完成**: `out/sft_qwen/final_model` 已就绪
5. ✅ **修复 Qwen DPO OOM**: 添加 `del logits` 释放中间张量，batch_size 改为 2

### ~~Phase 2: 执行实验~~ ✅ 已完成

6. ✅ **DeepSleep DPO 6组**: 全部完成，结果见 2.7 节
7. ✅ **Qwen DPO 6组**: 全部完成
8. ✅ **收集全部12组 report.json**

### ~~Phase 3: 统计分析~~ ✅ 已完成

9. ✅ **2²全因子 ANOVA 脚本**: `scripts/analysis/analyze_factorial.py`
   - 读取12组 `/root/blockdata/dpo_exp/*/report.json`
   - 计算 ANOVA 表（因素A, B, A×B, 误差）
   - 生成主效应图 (Main Effects Plot)
   - 生成交互效应图 (Interaction Plot)
   - 残差诊断（正态概率图、残差vs拟合值图）
   - 输出 F 值、p 值、显著性判断
   - 生成 Fig 0-8 共 9 张图 + Markdown 分析报告

10. ✅ **Benchmark acc_norm 多因素方差分析**: `scripts/analysis/plot_benchmark_anova.py`
    - 无重复双因素 ANOVA: Model(8) × Benchmark(5)
    - 手动计算 (SS, MS, F, p, η²) + statsmodels 验证
    - Tukey HSD 事后检验
    - 分组柱状图 (8模型×5benchmark, 最高分黑色线框标注)

11. ✅ **生成质量评估**: `scripts/evaluation/eval_quality.py`
    - 30条统一测试 prompt (6类别混合)
    - 4个DPO模型生成回复
    - DeepSeek V4 按10维度打分 (1-10分)
    - One-way ANOVA + 配对t检验 + 显著性括号柱状图

### ~~Phase 4: lm-evaluation-harness 评估~~ ✅ 已完成

12. ✅ **模型准备**: 8个模型全部转换为 HF 格式
13. ✅ **Benchmark 评估**: 8模型 × 5 benchmark (PubMedQA/MedQA/ARC-Easy/PIQA/OpenBookQA)
14. ✅ **雷达图**: 8模型5维雷达图 (`fig_benchmark_radar.png`)
15. ✅ **数据表**: `docs/figures/benchmark_results.csv`

### Phase 5: 撰写报告 🔄 进行中

16. ✅ **训练曲线对比图**: Fig 1 (4面板) + Fig 1s (12条叠加)
17. ✅ **ANOVA 主效应图**: Fig 2
18. ✅ **交互效应图**: Fig 3
19. ✅ **ANOVA 表**: Fig 6 (Pareto) + 终端输出
20. ✅ **残差诊断图**: Fig 4 (Y1 log + Y3)
21. ✅ **雷达图**: `fig_benchmark_radar.png`
22. ✅ **Pipeline Waterfall 图**: `fig_pipeline_waterfall.png`
23. ✅ **生成质量柱状图**: `fig_quality_anova.png`
24. ✅ **Benchmark ANOVA 柱状图**: `fig_benchmark_anova_barplot.png`
25. ⬜ **撰写完整实验报告**: 按 4.2 节结构撰写

---

## 6. 时间规划

| 阶段 | 内容 | 状态 | 预计时间 |
|------|------|------|----------|
| Phase 1 | 脚本修复+创建 | ✅ 完成 | — |
| Phase 2 | 12组DPO实验 | ✅ 完成 | ~3h GPU |
| Phase 3 | ANOVA统计分析 | ✅ 完成 | 3-4h |
| Phase 4 | lm-eval评估 (8模型×5 benchmark) | ✅ 完成 | 3-4h |
| Phase 5 | 撰写报告 | 🔄 进行中 | 4-6h |
| **总计** | | | **~12h** |

> 作业截止日期: **6月28日**（第18周周日），目标6月5日前完成初稿。

---

## 7. 关键文件索引

| 文件 | 说明 |
|------|------|
| `trainer/train_dpo.py` | DeepSleep DPO 训练（含JSONL日志+accuracy+report） |
| `trainer/train_dpo_qwen.py` | Qwen DPO 训练（含内存优化del logits） |
| `scripts/run/exp/*.sh` | 12个独立实验启动脚本 |
| `configs/dpo.yaml` | DeepSleep DPO 配置 |
| `configs/dpo_qwen.yaml` | Qwen DPO 配置 |
| `dataset/lm_dataset.py` | DPODataset 数据集类 |
| `/root/blockdata/dpo_exp/*/report.json` | 12组实验结果 |
| `/root/dslm/lm-evaluation-harness/` | lm-eval 评估框架 |
| `scripts/eval/convert_to_hf.py` | DeepSleep .pth → HF格式转换 |
| `scripts/eval/run_benchmark.sh` | lm-eval 8模型×5 benchmark 评估脚本 |
| `scripts/eval/merge_results.py` | 合并增量benchmark结果到已有文件 |
| `scripts/eval/plot_radar.py` | 雷达图+CSV数据表生成 |
| `data/eval/benchmark_results/` | lm-eval 评估结果目录 |
| `data/eval/pubmedqa_local/` | PubMedQA 本地数据 (lm-eval自定义任务) |
