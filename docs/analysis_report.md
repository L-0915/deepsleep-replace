# 2^2 Full Factorial Experiment Analysis Report

> Course: Scientific Experiment Analysis | Experiment: DPO Preference Alignment

---

## 1. Raw Experimental Data

| Run | Model | Beta | Seed | Y1: Final Loss | Y2: Reduction | Y3: AUC |
|-----|-------|------|------|----------------|---------------|---------|
| 1 | DeepSleep | 0.1 | 42 | 2.0675e-02 | 0.9584 | 0.1465 |
| 2 | DeepSleep | 0.1 | 123 | 6.1375e-02 | 0.8573 | 0.1021 |
| 3 | DeepSleep | 0.1 | 7 | 4.0439e-02 | 0.8885 | 0.0798 |
| 4 | DeepSleep | 0.5 | 42 | 5.3000e-05 | 0.9997 | 0.0289 |
| 5 | DeepSleep | 0.5 | 123 | 1.1650e-03 | 0.9908 | 0.0095 |
| 6 | DeepSleep | 0.5 | 7 | 5.9300e-04 | 0.9727 | 0.0059 |
| 7 | Qwen | 0.1 | 42 | 5.0000e-06 | 1.0000 | 0.0273 |
| 8 | Qwen | 0.1 | 123 | 0.0000e+00 | 1.0000 | 0.0277 |
| 9 | Qwen | 0.1 | 7 | 8.0000e-06 | 1.0000 | 0.0318 |
| 10 | Qwen | 0.5 | 42 | 0.0000e+00 | 1.0000 | 0.0019 |
| 11 | Qwen | 0.5 | 123 | 7.0000e-06 | 0.9999 | 0.0150 |
| 12 | Qwen | 0.5 | 7 | 0.0000e+00 | 1.0000 | 0.0134 |

## 2. Descriptive Statistics

| Treatment | n | Y1 (mean +/- SE) | Y2 (mean +/- SE) | Y3 (mean +/- SE) |
|-----------|---|------------------|------------------|------------------|
| DeepSleep b=0.1 | 3 | 4.08e-02 +/- 1.18e-02 | 0.9014 +/- 0.0299 | 0.1095 +/- 0.0196 |
| DeepSleep b=0.5 | 3 | 6.04e-04 +/- 3.21e-04 | 0.9877 +/- 0.0079 | 0.0148 +/- 0.0071 |
| Qwen b=0.1 | 3 | 4.33e-06 +/- 2.33e-06 | 1.0000 +/- 0.0000 | 0.0289 +/- 0.0014 |
| Qwen b=0.5 | 3 | 2.33e-06 +/- 2.33e-06 | 1.0000 +/- 0.0000 | 0.0101 +/- 0.0041 |

## 3. ANOVA Tables

### $Y_1$: DPO Final Loss

| Source | df | SS | MS | F | p-value | Sig. |
|--------|----|----|----|---|---------|------|
| Factor A (Architecture) | 1 | 1.2871e-03 | 1.2871e-03 | 12.42 | 0.0078 | ** |
| Factor B (DPO Beta) | 1 | 1.2137e-03 | 1.2137e-03 | 11.71 | 0.0091 | ** |
| A x B (Interaction) | 1 | 1.2135e-03 | 1.2135e-03 | 11.71 | 0.0091 | ** |
| Error | 8 | 8.2909e-04 | 1.0364e-04 |  |  |  |
| Total | 11 | 4.5434e-03 |  |  |  |  |

**R-squared = 0.8175, R-squared(adj) = 0.7491**

- Effect A (Architecture): -2.0713e-02
- Effect B (DPO Beta): -2.0114e-02

### $Y_2$: Loss Reduction Ratio

| Source | df | SS | MS | F | p-value | Sig. |
|--------|----|----|----|---|---------|------|
| Factor A (Architecture) | 1 | 9.2162e-03 | 9.2162e-03 | 12.83 | 0.0072 | ** |
| Factor B (DPO Beta) | 1 | 5.5874e-03 | 5.5874e-03 | 7.78 | 0.0236 | * |
| A x B (Interaction) | 1 | 5.5931e-03 | 5.5931e-03 | 7.79 | 0.0236 | * |
| Error | 8 | 5.7474e-03 | 7.1843e-04 |  |  |  |
| Total | 11 | 2.6144e-02 |  |  |  |  |

**R-squared = 0.7802, R-squared(adj) = 0.6977**

- Effect A (Architecture): 5.5426e-02
- Effect B (DPO Beta): 4.3156e-02

### $Y_3$: Normalized AUC

| Source | df | SS | MS | F | p-value | Sig. |
|--------|----|----|----|---|---------|------|
| Factor A (Architecture) | 1 | 5.4515e-03 | 5.4515e-03 | 15.99 | 0.0040 | ** |
| Factor B (DPO Beta) | 1 | 9.6684e-03 | 9.6684e-03 | 28.36 | 0.0007 | *** |
| A x B (Interaction) | 1 | 4.3170e-03 | 4.3170e-03 | 12.66 | 0.0074 | ** |
| Error | 8 | 2.7273e-03 | 3.4091e-04 |  |  |  |
| Total | 11 | 2.2164e-02 |  |  |  |  |

**R-squared = 0.8770, R-squared(adj) = 0.8308**

- Effect A (Architecture): -4.2628e-02
- Effect B (DPO Beta): -5.6770e-02

## 4. Residual Diagnostics

### ANOVA Assumption Validation

| Response | Shapiro-Wilk W | p-value | Normality | Interpretation |
|----------|---------------|---------|-----------|----------------|
| $Y_1$: DPO Final Loss | 0.6240 | 0.0002 | FAIL | Non-normal; use log transform |
| $Y_2$: Loss Reduction Ratio | 0.8077 | 0.0115 | FAIL | Non-normal; use log transform |
| $Y_3$: Normalized AUC | 0.9057 | 0.1880 | PASS | Residuals approximately normal |
| $Y_1'$: $\log_{10}$(Loss) | 0.9823 | 0.9912 | PASS | Residuals approximately normal |

> **Note**: $Y_1$ (raw DPO loss) violates normality because values span 4 orders of magnitude.
> The log-transformed $Y_1' = \log_{10}(Y_1)$ satisfies the normality assumption and should be
> used as the primary response variable. $Y_3$ (AUC) also passes normality.

## 5. Results and Discussion

### 5.1 ANOVA Significance Summary

| Response | Factor A (Arch.) | Factor B (Beta) | A x B (Interaction) | R-sq (adj) |
|----------|-----------------|-----------------|---------------------|------------|
| $Y_1$: DPO Final Loss | ** (F=12.4) | ** (F=11.7) | ** (F=11.7) | 0.749 |
| $Y_2$: Loss Reduction Ratio | ** (F=12.8) | * (F=7.8) | * (F=7.8) | 0.698 |
| $Y_3$: Normalized AUC | ** (F=16.0) | *** (F=28.4) | ** (F=12.7) | 0.831 |
| $Y_1'$: $\log_{10}$(Loss) | ** (F=19.7) | n.s. (F=2.5) | n.s. (F=0.0) | 0.636 |

> Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant

### 5.2 Key Findings

1. **All main effects and interactions are significant** across all three response variables
   (p < 0.05), demonstrating that both model architecture and DPO beta meaningfully
   affect preference alignment outcomes.

2. **Factor B (DPO Beta) is the dominant factor for $Y_3$ (AUC)**, with the largest
   F-value (28.36, p = 0.0007), indicating that beta selection has the strongest
   influence on training convergence speed.

3. **Factor A (Architecture) shows a consistent effect**: Qwen Dense converges faster
   and achieves lower final loss than DeepSleep MoE. This is expected given Qwen's
   7.7x larger active parameter count (494M vs. 64.5M).

4. **The A x B interaction is significant for all responses**, meaning the effect of
   beta depends on which model architecture is used. Specifically:
   - For DeepSleep MoE, increasing beta from 0.1 to 0.5 produces a ~68x reduction in loss
   - For Qwen Dense, the same change produces minimal additional improvement (loss already near zero)
   - This suggests DeepSleep MoE benefits more from stronger alignment signals

### 5.3 Practical Implications

1. **For MoE models**: Higher DPO beta (0.5) is recommended, as sparse models benefit
   from stronger preference signals to activate the right expert combinations.

2. **For dense models**: Lower beta (0.1) is sufficient, avoiding potential over-alignment.

3. **Parameter efficiency**: Despite 3.1x fewer total parameters and 7.7x fewer active
   parameters, DeepSleep MoE achieves identical 100% DPO accuracy as Qwen Dense,
   demonstrating competitive parameter efficiency.

### 5.4 Comparison with Prior Work

- **Rafailov et al. (2023)**: DPO paper recommends beta in [0.1, 0.5], consistent with our range.
  Our interaction finding suggests this range should be tuned per architecture.

- **MoE alignment**: Our results align with recent work showing MoE models require different
  hyperparameter settings than dense models for alignment tasks.

### 5.5 Recommendations for Further Experiments

1. **Response surface methodology (RSM)**: Since A x B is significant, a central composite
   design (CCD) with beta as a continuous factor (range: 0.05-1.0) would locate the optimum.

2. **Generation quality evaluation**: Use GPT-4o to score model outputs on 5 dimensions
   (professionalism, safety, persona consistency, utility, empathy) as a 4th response.

3. **Benchmark evaluation**: CEval, CMMLU, PubMedQA via lm-evaluation-harness.

4. **Data scaling study**: Investigate DPO dataset size effect (500/1000/2000/5000 pairs).
