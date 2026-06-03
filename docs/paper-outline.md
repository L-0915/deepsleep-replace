# 论文大纲：基于2²全因子设计的域特定大语言模型DPO对齐实验研究

> 《科学实验分析》课程大作业
> 2026年5月28日

## 基本信息

- **题目**: 基于2²全因子设计的域特定大语言模型DPO对齐实验研究——以睡眠健康领域为例
- **字数目标**: 至少8000字
- **语言**: 中文
- **格式**: 四段式（背景→实验设计→实验结果和处理→结果与讨论）+ 参考文献 + 附录

---

## 1. 背景（~1200字）

### 1.1 研究背景（~500字）

- LLM在通用领域取得突破性进展（GPT-4, LLaMA, Qwen等），参数规模从数十亿到数万亿
- 垂直领域应用的兴起：医疗健康是LLM落地的关键赛道之一
  - Med-PaLM (Singhal et al., Nature 2023) 首次通过USMLE风格问题
  - Med-PaLM 2 (Nature Medicine 2024) 达到专家水平
  - 华佗GPT (HuatuoGPT, EMNLP 2023) 中文医疗LLM
  - ChatDoctor (Li et al., 2023) 基于LLaMA的医学微调
- 睡眠健康作为垂直领域的独特价值：
  - 全球约30%人口受睡眠障碍影响
  - SleepFM (Stanford Medicine, 2026) 首个利用睡眠数据预测100+种健康风险的AI
  - PH-LLM (Nature Medicine, 2025) 基于Gemini微调的个人健康LLM
  - JMIR (2026) 研究表明AI聊天机器人可有效改善睡眠
- 现有医疗LLM存在两个核心未解问题：
  1. 架构选择：MoE vs Dense在域特定对齐中的差异尚缺乏系统性研究
  2. 对齐调参：DPO的Beta参数最优值是否因架构而异

### 1.2 模型架构：MoE vs Dense（~300字）

- **MoE（稀疏混合专家）** 的原理和优势：
  - GShard (Lepikhin et al., ICLR 2021) 首次扩展到6000亿参数
  - Switch Transformer (Fedus et al., JMLR 2021) top-1路由简化
  - Mixtral 8x7B (Jiang et al., 2024) 130亿活跃参数达到Llama-2 70B级别性能
  - Qwen2.5-MoE (Alibaba, 2024) 大规模MoE在20万亿tokens上预训练
- **MoE对齐的挑战**：
  - 专家坍塌 (Expert Collapse)：路由收敛到重复使用相同专家
  - RL训练灾难性崩溃 (arXiv:2510.11370)
  - 路由不稳定和负载不均衡
- **Dense架构** 的稳定性优势，但参数效率较低

### 1.3 DPO对齐方法（~200字）

- RLHF的局限：需要训练奖励模型+强化学习，训练不稳定
- DPO (Rafailov et al., NeurIPS 2023) 直接从偏好数据优化策略
- DPO损失函数及Beta参数物理含义
  - Beta越高 → 更强的偏好信号 → 但可能过拟合
  - Beta越低 → 更保守 → 更接近参考策略
  - 常用范围 0.1-0.5
- 后续改进：IPO, KTO, ORPO, beta-DPO (NeurIPS 2024)

### 1.4 实验目的（~100字）

### 1.5 实验响应（~150字）

- Y₁: DPO训练最终收敛损失
- Y₂: 损失下降比
- Y₃: 归一化AUC
- 补充：5 Benchmark + 10维度生成质量评分

---

## 2. 实验设计（~1200字）

### 2.1 设计方法选择（~300字）

- 2²全因子设计的理由
- 与正交设计、响应曲面、单因素法的对比
- ANOVA在ML超参数实验中的应用

### 2.2 因素与水平（~300字）

- 因素A：模型架构 (DeepSleep MoE 199M/64.5M active vs Qwen2.5-0.5B Dense 494M)
  - DeepSleep架构详述
- 因素B：DPO Beta (0.1 vs 0.5)

### 2.3 实验矩阵与重复（~200字）

- 4处理×3重复=12 runs
- 编码表

### 2.4 控制变量（~200字）

### 2.5 数据构建（~200字）

- SFT 6类 + DPO 6类×6反面风格
- LLM辅助两步分离生成流程

---

## 3. 实验结果和处理（~3000字）

### 3.1 原始数据与描述统计（~300字）

- 12组原始数据表 + 描述统计表
- [图0] fig0_summary_table.png
- [图1] fig1_training_curves.png

### 3.2 ANOVA 方差分析（~500字）

- 线性统计模型: Y = μ + α·A + β·B + (αβ)·A×B + ε
- Y₁/Y₂/Y₃ 三张ANOVA表
- 显著性判断

### 3.3 主效应分析（~350字）

- [图2] fig2_main_effects.png
- 因素A和B的效应量

### 3.4 交互效应分析（~450字）

- [图3] fig3_interaction.png
- [图6] fig6_pareto.png
- [图7] fig7_heatmap.png
- 定量解释：MoE对beta更敏感

### 3.5 残差诊断（~350字）

- Shapiro-Wilk检验结果
- log变换处理
- [图4a] fig4_residuals_Y1_loss.png
- [图4b] fig4_residuals_Y3_auc.png

### 3.6 Benchmark 多因素分析（~500字）

- 8模型×5 benchmark 无重复双因素ANOVA
- [图9] fig_benchmark_anova_barplot.png
- [图10] fig_benchmark_radar.png
- Tukey HSD事后检验
- 结论：预训练基础 > DPO微调

### 3.7 生成质量评估（~350字）

- 30 prompt × 4模型 × 10维度
- [图11] fig_quality_anova.png
- One-way ANOVA

---

## 4. 结果与讨论（~1500字）

### 4.1 主效应结论（~300字）

### 4.2 交互效应与架构敏感性（~350字）

- MoE需要更强对齐信号的理论解释
- 与MoE-DPO (arXiv:2510.08256) 的联系

### 4.3 最优实验条件（~200字）

- [图12] fig_pipeline_waterfall.png

### 4.4 与前人结果比较（~300字）

- Rafailov et al. (2023) Beta∈[0.1, 0.5]
- Med-PaLM/HuatuoGPT的对齐策略
- 本研究的创新发现

### 4.5 进一步实验建议（~200字）

- RSM/CCD连续化Beta
- 数据规模效应

### 4.6 应用展示（~150字）

- DeepSleep Chat简要介绍

---

## 5. 参考文献（~20条）

核心文献列表见论文正文。

---

## 6. 附录

- [图1s] fig1s_overlay.png 训练曲线叠加
- [图5] fig5_boxplots.png
- [图8] fig8_convergence.png
- 生成样例对比

---

## 图表索引

| 编号 | 文件 | 位置 | 说明 |
|------|------|------|------|
| 图1 | fig0_summary_table.png | 3.1 | 实验数据汇总表 |
| 图2 | fig1_training_curves.png | 3.1/附录 | 训练曲线4面板 |
| 图3 | fig2_main_effects.png | 3.3 | 主效应图 |
| 图4 | fig3_interaction.png | 3.4 | 交互效应图 |
| 图5 | fig6_pareto.png | 3.4 | Pareto效应图 |
| 图6 | fig7_heatmap.png | 3.4 | 热力图 |
| 图7a | fig4_residuals_Y1_loss.png | 3.5 | Y1残差诊断 |
| 图7b | fig4_residuals_Y3_auc.png | 3.5 | Y3残差诊断 |
| 图8 | fig_benchmark_anova_barplot.png | 3.6 | Benchmark ANOVA柱状图 |
| 图9 | fig_benchmark_radar.png | 3.6 | 雷达图 |
| 图10 | fig_quality_anova.png | 3.7 | 生成质量柱状图 |
| 图11 | fig_pipeline_waterfall.png | 4.3 | Pipeline Waterfall图 |
| 附1 | fig1s_overlay.png | 附录 | 训练曲线叠加 |
| 附2 | fig5_boxplots.png | 附录 | 箱线图 |
| 附3 | fig8_convergence.png | 附录 | 收敛分析 |
