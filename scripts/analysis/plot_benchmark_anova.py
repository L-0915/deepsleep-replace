#!/usr/bin/env python3
"""
Two-Way ANOVA on acc_norm benchmark scores + grouped bar chart.

Design:  Two-way layout without replication (1 obs per cell)
Factors: Model (8 levels) × Benchmark (5 levels)
Response: acc_norm score (%)

Since there is exactly one observation per (Model, Benchmark) cell,
the interaction term and the error term are confounded.  We therefore
use the standard approach for a Randomized Complete Block Design:
  • residual from the main-effects model = Model × Benchmark interaction
  • F-tests for Model and Benchmark use this residual as the denominator

Output:  docs/figures/fig_benchmark_anova_barplot.png / .pdf
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
FIGURE_DIR = "/root/dslm/deepsleep/docs/figures"
CSV_PATH = os.path.join(FIGURE_DIR, "benchmark_results.csv")

# ── Journal-style rcParams (match existing analyze_factorial.py) ──────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.edgecolor": "0.85",
    "legend.fancybox": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
})

# ── Color palette (colorblind-safe, grouped by model family) ──────────────
COLORS = {
    "DeepSleep-Base":        "#7B68EE",
    "DeepSleep-DPO(β=0.1)": "#6A5ACD",
    "DeepSleep-DPO(β=0.5)": "#483D8B",
    "MiniMind-3":            "#E69F00",
    "Medical-GPT2":          "#D55E00",
    "Qwen2.5-Base":          "#56B4E9",
    "Qwen2.5-DPO(β=0.1)":   "#2B86BA",
    "Qwen2.5-DPO(β=0.5)":   "#0E5C8A",
}

HATCHES = {
    "DeepSleep-Base":        "",
    "DeepSleep-DPO(β=0.1)": "//",
    "DeepSleep-DPO(β=0.5)": "xx",
    "MiniMind-3":            "",
    "Medical-GPT2":          "",
    "Qwen2.5-Base":          "",
    "Qwen2.5-DPO(β=0.1)":   "//",
    "Qwen2.5-DPO(β=0.5)":   "xx",
}


def sig_mark(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(CSV_PATH)
    benchmarks = [
        ("PubMedQA",  "PubMedQA acc"),
        ("MedQA",     "MedQA acc_norm"),
        ("ARC-Easy",  "ARC-Easy acc_norm"),
        ("PIQA",      "PIQA acc_norm"),
        ("OpenBookQA","OpenBookQA acc_norm"),
    ]
    records = []
    for _, row in df.iterrows():
        for bname, col in benchmarks:
            records.append({
                "Model": row["Model"],
                "Benchmark": bname,
                "Score": row[col],
            })
    long_df = pd.DataFrame(records)
    return long_df, df, [b[0] for b in benchmarks], [b[1] for b in benchmarks]


# ═══════════════════════════════════════════════════════════════════════════
# ANOVA — manual computation (transparent, no NaN issues)
# ═══════════════════════════════════════════════════════════════════════════

def do_anova(long_df):
    """
    Two-way ANOVA without replication (Randomized Complete Block Design).

    With 1 obs/cell the interaction is confounded with error, so:
      SS_Error = SS_Total - SS_Model - SS_Benchmark
    and F = MS_factor / MS_Error tests each factor against the interaction.
    """
    y = long_df["Score"].values.astype(float)
    n = len(y)

    a_levels = sorted(long_df["Model"].unique())
    b_levels = sorted(long_df["Benchmark"].unique())
    a = len(a_levels)   # 8 models
    b = len(b_levels)   # 5 benchmarks

    y_bar = np.mean(y)

    # ── Marginal & cell means ──────────────────────────────────────────────
    model_means = long_df.groupby("Model")["Score"].mean()
    bench_means = long_df.groupby("Benchmark")["Score"].mean()

    # ── Sums of squares ────────────────────────────────────────────────────
    SS_Total = np.sum((y - y_bar) ** 2)
    SS_Model = b * np.sum((model_means.values - y_bar) ** 2)
    SS_Bench = a * np.sum((bench_means.values - y_bar) ** 2)
    SS_Error = SS_Total - SS_Model - SS_Bench   # = Model×Benchmark interaction

    # ── Degrees of freedom ─────────────────────────────────────────────────
    df_Model = a - 1          # 7
    df_Bench = b - 1          # 4
    df_Error = (a - 1) * (b - 1)  # 28  (= interaction df)
    df_Total = n - 1          # 39

    # ── Mean squares ───────────────────────────────────────────────────────
    MS_Model = SS_Model / df_Model
    MS_Bench = SS_Bench / df_Bench
    MS_Error = SS_Error / df_Error

    # ── F-values & p-values ────────────────────────────────────────────────
    F_Model = MS_Model / MS_Error
    F_Bench = MS_Bench / MS_Error
    p_Model = 1.0 - stats.f.cdf(F_Model, df_Model, df_Error)
    p_Bench = 1.0 - stats.f.cdf(F_Bench, df_Bench, df_Error)

    # ── R-squared ──────────────────────────────────────────────────────────
    R2 = 1.0 - SS_Error / SS_Total
    R2_adj = 1.0 - (SS_Error / df_Error) / (SS_Total / df_Total)

    # ── Effect sizes (η²) ─────────────────────────────────────────────────
    eta2_Model = SS_Model / SS_Total
    eta2_Bench = SS_Bench / SS_Total

    # ── Print ANOVA table ──────────────────────────────────────────────────
    print("=" * 90)
    print("  Two-Way ANOVA (Model × Benchmark)  —  acc_norm scores, no replication")
    print("  Error term = Model × Benchmark interaction (confounded with pure error)")
    print("=" * 90)
    print()
    hdr = f"{'Source':<25} {'df':>4}  {'SS':>12}  {'MS':>12}  {'F':>10}  {'p-value':>12}  {'Sig.':>5}"
    print(hdr)
    print("-" * 90)
    rows = [
        ("Model",    df_Model, SS_Model, MS_Model, F_Model, p_Model),
        ("Benchmark",df_Bench, SS_Bench, MS_Bench, F_Bench, p_Bench),
        ("Error (= M×B)", df_Error, SS_Error, MS_Error, None, None),
        ("Total",    df_Total, SS_Total, None,     None,    None),
    ]
    for name, df_v, ss, ms, f_v, p_v in rows:
        ss_s = f"{ss:>12.4f}" if ss is not None else ""
        ms_s = f"{ms:>12.4f}" if ms is not None else ""
        f_s  = f"{f_v:>10.2f}" if f_v is not None else ""
        p_s  = f"{p_v:>12.6f}" if p_v is not None else ""
        sig  = sig_mark(p_v) if p_v is not None else ""
        print(f"{name:<25} {df_v:>4}  {ss_s}  {ms_s}  {f_s}  {p_s}  {sig:>5}")
    print("-" * 90)
    print(f"  R² = {R2:.4f}   R²(adj) = {R2_adj:.4f}")
    print(f"  η²(Model) = {eta2_Model:.4f}  ({eta2_Model*100:.1f}% variance)")
    print(f"  η²(Benchmark) = {eta2_Bench:.4f}  ({eta2_Bench*100:.1f}% variance)")
    print(f"  η²(Error) = {1-eta2_Model-eta2_Bench:.4f}  ({(1-eta2_Model-eta2_Bench)*100:.1f}% variance)")
    print("=" * 90)
    print()

    # ── Verify with statsmodels (should match) ────────────────────────────
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    lm = ols("Score ~ C(Model) + C(Benchmark)", data=long_df).fit()
    aov_sm = anova_lm(lm, typ=2)
    print("  [Verification] statsmodels Type-II results:")
    print(f"    F(Model)    = {aov_sm.loc['C(Model)', 'F']:.4f}   p = {aov_sm.loc['C(Model)', 'PR(>F)']:.6f}")
    print(f"    F(Benchmark)= {aov_sm.loc['C(Benchmark)', 'F']:.4f}   p = {aov_sm.loc['C(Benchmark)', 'PR(>F)']:.6f}")
    print(f"    (Manual ↔ statsmodels should agree.)")
    print()

    return {
        "F_Model": F_Model, "p_Model": p_Model,
        "F_Bench": F_Bench, "p_Bench": p_Bench,
        "R2": R2, "R2_adj": R2_adj,
        "eta2_Model": eta2_Model, "eta2_Bench": eta2_Bench,
        "SS_Model": SS_Model, "SS_Bench": SS_Bench,
        "SS_Error": SS_Error, "SS_Total": SS_Total,
        "df_Model": df_Model, "df_Bench": df_Bench,
        "df_Error": df_Error,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Post-hoc: Tukey HSD
# ═══════════════════════════════════════════════════════════════════════════

def posthoc_tukey(long_df):
    """Tukey HSD pairwise comparisons for Model factor."""
    print("── Post-hoc: Tukey HSD (Model pairwise) ──────────────────────")
    tukey = pairwise_tukeyhsd(
        endog=long_df["Score"].values,
        groups=long_df["Model"].values,
        alpha=0.05,
    )
    print(tukey.summary())
    print()

    # Also show model means ranked
    print("  Model mean acc_norm (ranked):")
    model_means = long_df.groupby("Model")["Score"].mean().sort_values(ascending=False)
    for m, v in model_means.items():
        print(f"    {m:<30} {v:>8.2f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_grouped_bar(df_wide, benchmark_names, col_names, anova):
    """Grouped bar chart: x = Benchmarks, bars = Models."""
    models = df_wide["Model"].tolist()
    n_models = len(models)
    n_benchmarks = len(benchmark_names)

    # Build score matrix (model × benchmark)
    scores = np.zeros((n_models, n_benchmarks))
    for i, model in enumerate(models):
        for j, col in enumerate(col_names):
            scores[i, j] = df_wide.loc[df_wide["Model"] == model, col].values[0]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6.5))

    x = np.arange(n_benchmarks)
    group_width = 0.78
    bar_width = group_width / n_models

    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, scores[i], bar_width,
            label=model,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.6,
            hatch=HATCHES.get(model, ""),
            zorder=3,
        )
        # Value labels on top of bars
        for bar, val in zip(bars, scores[i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.6,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=5.5, fontweight="bold",
                rotation=60,
            )

    # ── Highlight best model per benchmark with thin black outline ─────────
    for j in range(n_benchmarks):
        best_i = int(np.argmax(scores[:, j]))
        offset = (best_i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x[j] + offset, scores[best_i, j], bar_width,
            fill=False, edgecolor="black", linewidth=0.8,
            zorder=4,
        )

    # ── Axes formatting ────────────────────────────────────────────────────
    ax.set_xlabel("Benchmark Task", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("acc_norm Score (%)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_names, fontsize=11)

    y_max = scores.max()
    ax.set_ylim(0, y_max * 1.22)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # ── Legend (grouped by family) ─────────────────────────────────────────
    ds_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], edgecolor="white", hatch=HATCHES[m])
        for m in models if m.startswith("DeepSleep")
    ]
    ds_labels = [m for m in models if m.startswith("DeepSleep")]

    base_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], edgecolor="white")
        for m in models if m in ("MiniMind-3", "Medical-GPT2")
    ]
    base_labels = [m for m in models if m in ("MiniMind-3", "Medical-GPT2")]

    qw_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], edgecolor="white", hatch=HATCHES[m])
        for m in models if m.startswith("Qwen")
    ]
    qw_labels = [m for m in models if m.startswith("Qwen")]

    leg = ax.legend(
        ds_handles + base_handles + qw_handles,
        ds_labels + base_labels + qw_labels,
        loc="upper left",
        ncol=1,
        fontsize=8,
        framealpha=0.5,
        edgecolor="0.8",
    )

    ax.set_title(
        "Multi-Factor ANOVA: Model × Benchmark  (acc_norm)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # ── ANOVA significance annotation (PNG only) ─────────────────────────
    p_m = anova["p_Model"]
    p_b = anova["p_Bench"]
    anova_text = (
        f"Two-Way ANOVA (α = 0.05):\n"
        f"  Model:     F({anova['df_Model']},{anova['df_Error']}) = {anova['F_Model']:.2f}"
        f"  p = {p_m:.2e} {sig_mark(p_m)}\n"
        f"  Benchmark: F({anova['df_Bench']},{anova['df_Error']}) = {anova['F_Bench']:.2f}"
        f"  p = {p_b:.2e} {sig_mark(p_b)}\n"
        f"  R² = {anova['R2']:.4f}  R²(adj) = {anova['R2_adj']:.4f}"
    )

    fig.tight_layout()

    # Save PNG (no ANOVA annotation)
    png_path = os.path.join(FIGURE_DIR, "fig_benchmark_anova_barplot.png")
    fig.savefig(png_path)
    print(f"  Saved: {png_path}")

    # Save PDF (with ANOVA annotation)
    ax.text(
        0.98, 0.96, anova_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                  edgecolor="0.7", alpha=0.92),
    )
    pdf_path = os.path.join(FIGURE_DIR, "fig_benchmark_anova_barplot.pdf")
    fig.savefig(pdf_path)
    print(f"  Saved: {pdf_path}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Benchmark acc_norm — Multi-Factor ANOVA + Bar Chart")
    print("=" * 60)

    # 1. Load
    long_df, df_wide, benchmark_names, col_names = load_data()
    print(f"  Data: {len(df_wide)} models × {len(benchmark_names)} benchmarks"
          f" = {len(long_df)} observations\n")

    # 2. Descriptive stats
    print("── Descriptive Statistics (acc_norm, %) ────────────────────────")
    pivot = long_df.pivot(index="Model", columns="Benchmark", values="Score")
    print(pivot.to_string(float_format="%.2f"))
    print()

    # 3. ANOVA
    anova = do_anova(long_df)

    # 4. Post-hoc
    posthoc_tukey(long_df)

    # 5. Chart
    print("[Plot] Generating grouped bar chart...")
    plot_grouped_bar(df_wide, benchmark_names, col_names, anova)

    print("[Done]")


if __name__ == "__main__":
    main()
