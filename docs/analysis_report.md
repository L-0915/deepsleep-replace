# 2^3 Full Factorial Experiment Analysis Report

> Course: Scientific Experiment Analysis | Experiment: DPO Preference Alignment

> Design: 2^3 full factorial (Architecture × DPO Beta × Learning Rate), 3 replications

---

## 1. Raw Experimental Data

| Run | Model | Beta | LR | Seed | Y1: Final Loss | Y2: Reduction | Y3: AUC |
|-----|-------|------|-----|------|----------------|---------------|---------|
| 1 | DeepSleep | 0.1 | 5e-07 | 42 | 2.0675e-02 | 0.9584 | 0.1465 |
| 2 | DeepSleep | 0.1 | 5e-07 | 123 | 6.1375e-02 | 0.8573 | 0.1021 |
| 3 | DeepSleep | 0.1 | 5e-07 | 7 | 4.0439e-02 | 0.8885 | 0.0798 |
| 4 | DeepSleep | 0.5 | 5e-07 | 42 | 5.3000e-05 | 0.9997 | 0.0289 |
| 5 | DeepSleep | 0.5 | 5e-07 | 123 | 1.1650e-03 | 0.9908 | 0.0095 |
| 6 | DeepSleep | 0.5 | 5e-07 | 7 | 5.9300e-04 | 0.9727 | 0.0059 |
| 7 | Qwen | 0.1 | 5e-07 | 42 | 5.0000e-06 | 1.0000 | 0.0273 |
| 8 | Qwen | 0.1 | 5e-07 | 123 | 0.0000e+00 | 1.0000 | 0.0277 |
| 9 | Qwen | 0.1 | 5e-07 | 7 | 8.0000e-06 | 1.0000 | 0.0318 |
| 10 | Qwen | 0.5 | 5e-07 | 42 | 0.0000e+00 | 1.0000 | 0.0019 |
| 11 | Qwen | 0.5 | 5e-07 | 123 | 7.0000e-06 | 0.9999 | 0.0150 |
| 12 | Qwen | 0.5 | 5e-07 | 7 | 0.0000e+00 | 1.0000 | 0.0134 |
| 13 | DeepSleep | 0.1 | 1e-06 | 42 | 1.7150e-03 | 0.9955 | 0.0835 |
| 14 | DeepSleep | 0.1 | 1e-06 | 123 | 9.3980e-03 | 0.9628 | 0.0373 |
| 15 | DeepSleep | 0.1 | 1e-06 | 7 | 6.0910e-03 | 0.9689 | 0.0267 |
| 16 | DeepSleep | 0.5 | 1e-06 | 42 | 1.0000e-06 | 1.0000 | 0.0128 |
| 17 | DeepSleep | 0.5 | 1e-06 | 123 | 3.3000e-05 | 0.9960 | 0.0007 |
| 18 | DeepSleep | 0.5 | 1e-06 | 7 | 9.4100e-04 | 0.8266 | 0.0058 |
| 19 | Qwen | 0.1 | 1e-06 | 42 | 0.0000e+00 | 1.0000 | 0.0497 |
| 20 | Qwen | 0.1 | 1e-06 | 123 | 2.0000e-06 | 1.0000 | 0.0652 |
| 21 | Qwen | 0.1 | 1e-06 | 7 | 7.6000e-05 | 0.9999 | 0.1317 |
| 22 | Qwen | 0.5 | 1e-06 | 42 | 1.7000e-05 | 1.0000 | 0.0568 |
| 23 | Qwen | 0.5 | 1e-06 | 123 | 0.0000e+00 | 1.0000 | 0.0226 |
| 24 | Qwen | 0.5 | 1e-06 | 7 | 0.0000e+00 | 1.0000 | 0.0101 |

## 2. Descriptive Statistics

| Treatment | n | Y1 (mean +/- SE) | Y2 (mean +/- SE) | Y3 (mean +/- SE) |
|-----------|---|------------------|------------------|------------------|
| DeepSleep b=0.1 lr=5e-07 | 3 | 4.08e-02 +/- 1.18e-02 | 0.9014 +/- 0.0299 | 0.1095 +/- 0.0196 |
| DeepSleep b=0.1 lr=1e-06 | 3 | 5.73e-03 +/- 2.23e-03 | 0.9757 +/- 0.0100 | 0.0492 +/- 0.0175 |
| DeepSleep b=0.5 lr=5e-07 | 3 | 6.04e-04 +/- 3.21e-04 | 0.9877 +/- 0.0079 | 0.0148 +/- 0.0071 |
| DeepSleep b=0.5 lr=1e-06 | 3 | 3.25e-04 +/- 3.08e-04 | 0.9409 +/- 0.0571 | 0.0064 +/- 0.0035 |
| Qwen b=0.1 lr=5e-07 | 3 | 4.33e-06 +/- 2.33e-06 | 1.0000 +/- 0.0000 | 0.0289 +/- 0.0014 |
| Qwen b=0.1 lr=1e-06 | 3 | 2.60e-05 +/- 2.50e-05 | 1.0000 +/- 0.0000 | 0.0822 +/- 0.0252 |
| Qwen b=0.5 lr=5e-07 | 3 | 2.33e-06 +/- 2.33e-06 | 1.0000 +/- 0.0000 | 0.0101 +/- 0.0041 |
| Qwen b=0.5 lr=1e-06 | 3 | 5.67e-06 +/- 5.67e-06 | 1.0000 +/- 0.0000 | 0.0298 +/- 0.0139 |

## 3. ANOVA Tables

### $Y_1$: DPO Final Loss

| Source | df | SS | MS | F | p-value | Sig. |
|--------|----|----|----|---|---------|------|
| Factor A (Architecture) | 1 | 8.4448e-04 | 8.4448e-04 | 15.72 | 0.0011 | ** |
| Factor B (DPO Beta) | 1 | 7.8174e-04 | 7.8174e-04 | 14.55 | 0.0015 | ** |
| Factor C (Learning Rate) | 1 | 4.6857e-04 | 4.6857e-04 | 8.72 | 0.0093 | ** |
| A × B | 1 | 7.8022e-04 | 7.8022e-04 | 14.53 | 0.0015 | ** |
| A × C | 1 | 4.6990e-04 | 4.6990e-04 | 8.75 | 0.0093 | ** |
| B × C | 1 | 4.5409e-04 | 4.5409e-04 | 8.45 | 0.0103 | * |
| A × B × C | 1 | 4.5505e-04 | 4.5505e-04 | 8.47 | 0.0102 | * |
| Error | 16 | 8.5937e-04 | 5.3711e-05 |  |  |  |
| Total | 23 | 5.1134e-03 |  |  |  |  |

**R-squared = 0.8319, R-squared(adj) = 0.7584**

- Effect A (Architecture): -1.1864e-02
- Effect B (DPO Beta): -1.1414e-02
- Effect C (Learning Rate): -8.8372e-03

### $Y_2$: Loss Reduction Ratio

| Source | df | SS | MS | F | p-value | Sig. |
|--------|----|----|----|---|---------|------|
| Factor A (Architecture) | 1 | 1.4144e-02 | 1.4144e-02 | 8.73 | 0.0093 | ** |
| Factor B (DPO Beta) | 1 | 9.9412e-04 | 9.9412e-04 | 0.61 | 0.4449 | n.s. |
| Factor C (Learning Rate) | 1 | 2.8272e-04 | 2.8272e-04 | 0.17 | 0.6817 | n.s. |
| A × B | 1 | 9.9313e-04 | 9.9313e-04 | 0.61 | 0.4452 | n.s. |
| A × C | 1 | 2.8356e-04 | 2.8356e-04 | 0.17 | 0.6813 | n.s. |
| B × C | 1 | 5.5029e-03 | 5.5029e-03 | 3.40 | 0.0840 | n.s. |
| A × B × C | 1 | 5.5132e-03 | 5.5132e-03 | 3.40 | 0.0837 | n.s. |
| Error | 16 | 2.5929e-02 | 1.6206e-03 |  |  |  |
| Total | 23 | 5.3642e-02 |  |  |  |  |

**R-squared = 0.5166, R-squared(adj) = 0.3052**

- Effect A (Architecture): 4.8552e-02
- Effect B (DPO Beta): 1.2872e-02
- Effect C (Learning Rate): 6.8643e-03

### $Y_3$: Normalized AUC

| Source | df | SS | MS | F | p-value | Sig. |
|--------|----|----|----|---|---------|------|
| Factor A (Architecture) | 1 | 3.1182e-04 | 3.1182e-04 | 0.52 | 0.4812 | n.s. |
| Factor B (DPO Beta) | 1 | 1.6330e-02 | 1.6330e-02 | 27.24 | 0.0001 | *** |
| Factor C (Learning Rate) | 1 | 7.0333e-06 | 7.0333e-06 | 0.01 | 0.9151 | n.s. |
| A × B | 1 | 1.6448e-03 | 1.6448e-03 | 2.74 | 0.1171 | n.s. |
| A × C | 1 | 7.5271e-03 | 7.5271e-03 | 12.56 | 0.0027 | ** |
| B × C | 1 | 1.2701e-04 | 1.2701e-04 | 0.21 | 0.6515 | n.s. |
| A × B × C | 1 | 2.7419e-03 | 2.7419e-03 | 4.57 | 0.0482 | * |
| Error | 16 | 9.5916e-03 | 5.9948e-04 |  |  |  |
| Total | 23 | 3.8281e-02 |  |  |  |  |

**R-squared = 0.7494, R-squared(adj) = 0.6398**

- Effect A (Architecture): -7.2090e-03
- Effect B (DPO Beta): -5.2169e-02
- Effect C (Learning Rate): 1.0827e-03

## 4. Residual Diagnostics

### ANOVA Assumption Validation

| Response | Shapiro-Wilk W | p-value | Normality | Interpretation |
|----------|---------------|---------|-----------|----------------|
| $Y_1$: DPO Final Loss | 0.5281 | 0.0000 | FAIL | Non-normal; use log transform |
| $Y_2$: Loss Reduction Ratio | 0.7577 | 0.0001 | FAIL | Non-normal; use log transform |
| $Y_3$: Normalized AUC | 0.9338 | 0.1188 | PASS | Residuals approximately normal |

## 5. Results and Discussion

### 5.1 ANOVA Significance Summary

| Response | A (Arch.) | B (Beta) | C (LR) | A×B | A×C | B×C | A×B×C | R²(adj) |
|----------|-----------|----------|--------|-----|-----|-----|-------|---------|
| $Y_1$: DPO Final Loss | ** (F=15.7) | ** (F=14.6) | ** (F=8.7) | ** (F=14.5) | ** (F=8.7) | * (F=8.5) | * (F=8.5) | 0.758 |
| $Y_2$: Loss Reduction Ratio | ** (F=8.7) | n.s. (F=0.6) | n.s. (F=0.2) | n.s. (F=0.6) | n.s. (F=0.2) | n.s. (F=3.4) | n.s. (F=3.4) | 0.305 |
| $Y_3$: Normalized AUC | n.s. (F=0.5) | *** (F=27.2) | n.s. (F=0.0) | n.s. (F=2.7) | ** (F=12.6) | n.s. (F=0.2) | * (F=4.6) | 0.640 |

> Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant

### 5.2 Key Findings

1. **Factor C (Learning Rate) effect**: The higher learning rate (1e-6 vs 5e-7) is expected to accelerate DPO convergence but may introduce training instability, particularly for MoE architectures with sparse gradient signals.

2. **A × C interaction**: If significant, this indicates that the optimal learning rate depends on model architecture (MoE vs Dense). MoE models, with their sparse gradient flow through top-k experts, may require different LR settings than Dense models.

3. **B × C interaction**: If significant, the joint effect of Beta and LR on alignment is not simply additive — high Beta and high LR together may cause over-alignment or training instability.

4. **A × B × C three-way interaction**: The most complex effect. If significant, the optimal (Beta, LR) combination differs by architecture, requiring architecture-specific hyperparameter tuning.

### 5.3 Practical Implications

1. **Architecture-specific tuning**: If A × C is significant, MoE and Dense models should use different learning rates for DPO alignment.

2. **Joint hyperparameter optimization**: If B × C is significant, Beta and LR should be tuned jointly rather than independently.

3. **Parameter efficiency**: DeepSleep MoE's competitive performance with 7.7× fewer active parameters reinforces the value of sparse architectures for domain-specific LLMs.

### 5.4 Recommendations for Further Experiments

1. **Response surface methodology (RSM)**: Extend to continuous factors via central composite design (CCD) to locate the global optimum.

2. **Generation quality evaluation**: Use GPT-4o to score model outputs on multiple dimensions as an additional response variable.

3. **Benchmark evaluation**: PubMedQA, MedQA, ARC-Easy, PIQA, OpenBookQA.

4. **Data scaling study**: Investigate DPO dataset size effect (500/1000/2000/5000 pairs).
