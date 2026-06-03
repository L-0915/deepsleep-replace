#!/usr/bin/env python3
"""
2^3 Full Factorial ANOVA Analysis for DeepSleep DPO Experiments.

Publication-quality statistical analysis and visualization.
All figures follow Nature/Science journal style guidelines.

Factors:
  A: Model Architecture (DeepSleep MoE vs Qwen Dense)
  B: DPO Beta (0.1 vs 0.5)
  C: Learning Rate (5e-7 vs 1e-6)

Design: 2^3 full factorial, 3 replications per treatment (24 total runs)
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_DIR = "/root/blockdata/dpo_exp"
FIGURE_DIR = "/root/dslm/deepsleep/docs/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Journal-style rcParams ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "mathtext.default": "regular",
})

# ── Nature/Science color palette (colorblind-friendly) ────────────────────
PAL = {
    "blue": "#0072B2",
    "red": "#D55E00",
    "teal": "#009E73",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "light_blue": "#56B4E9",
    "light_red": "#F4A582",
    "black": "#000000",
}

# 8-group color palette: Model (Blue/Red) × Beta (Light/Dark) × LR (solid/marker)
# For training curves and box plots
GROUP_COLORS = {
    "DS-0.1-lr1": PAL["light_blue"],
    "DS-0.5-lr1": PAL["blue"],
    "QW-0.1-lr1": PAL["light_red"],
    "QW-0.5-lr1": PAL["red"],
    "DS-0.1-lr2": "#66C2A5",   # light teal
    "DS-0.5-lr2": PAL["teal"],
    "QW-0.1-lr2": PAL["yellow"],
    "QW-0.5-lr2": PAL["purple"],
}

GROUP_ORDER = [
    "DS-0.1-lr1", "DS-0.1-lr2",
    "DS-0.5-lr1", "DS-0.5-lr2",
    "QW-0.1-lr1", "QW-0.1-lr2",
    "QW-0.5-lr1", "QW-0.5-lr2",
]

GROUP_LABELS = {
    "DS-0.1-lr1": "DS\nβ=0.1\nLR=5e-7",
    "DS-0.1-lr2": "DS\nβ=0.1\nLR=1e-6",
    "DS-0.5-lr1": "DS\nβ=0.5\nLR=5e-7",
    "DS-0.5-lr2": "DS\nβ=0.5\nLR=1e-6",
    "QW-0.1-lr1": "Qwen\nβ=0.1\nLR=5e-7",
    "QW-0.1-lr2": "Qwen\nβ=0.1\nLR=1e-6",
    "QW-0.5-lr1": "Qwen\nβ=0.5\nLR=5e-7",
    "QW-0.5-lr2": "Qwen\nβ=0.5\nLR=1e-6",
}

# ── Experiment metadata (24 runs) ─────────────────────────────────────────
# (group, model, beta, lr, seed, coded_A, coded_B, coded_C, dirname)
EXPERIMENTS = [
    # ── Existing 12 runs: LR = 5e-7 (C = -1) ──
    ("DS-0.1-lr1", "DeepSleep", 0.1, 5e-7, 42,  -1, -1, -1, "ds_b0.1_s42"),
    ("DS-0.1-lr1", "DeepSleep", 0.1, 5e-7, 123, -1, -1, -1, "ds_b0.1_s123"),
    ("DS-0.1-lr1", "DeepSleep", 0.1, 5e-7, 7,   -1, -1, -1, "ds_b0.1_s7"),
    ("DS-0.5-lr1", "DeepSleep", 0.5, 5e-7, 42,  -1, +1, -1, "ds_b0.5_s42"),
    ("DS-0.5-lr1", "DeepSleep", 0.5, 5e-7, 123, -1, +1, -1, "ds_b0.5_s123"),
    ("DS-0.5-lr1", "DeepSleep", 0.5, 5e-7, 7,   -1, +1, -1, "ds_b0.5_s7"),
    ("QW-0.1-lr1", "Qwen",      0.1, 5e-7, 42,  +1, -1, -1, "qwen_b0.1_s42"),
    ("QW-0.1-lr1", "Qwen",      0.1, 5e-7, 123, +1, -1, -1, "qwen_b0.1_s123"),
    ("QW-0.1-lr1", "Qwen",      0.1, 5e-7, 7,   +1, -1, -1, "qwen_b0.1_s7"),
    ("QW-0.5-lr1", "Qwen",      0.5, 5e-7, 42,  +1, +1, -1, "qwen_b0.5_s42"),
    ("QW-0.5-lr1", "Qwen",      0.5, 5e-7, 123, +1, +1, -1, "qwen_b0.5_s123"),
    ("QW-0.5-lr1", "Qwen",      0.5, 5e-7, 7,   +1, +1, -1, "qwen_b0.5_s7"),
    # ── New 12 runs: LR = 1e-6 (C = +1) ──
    ("DS-0.1-lr2", "DeepSleep", 0.1, 1e-6, 42,  -1, -1, +1, "ds_b0.1_lr1e-6_s42"),
    ("DS-0.1-lr2", "DeepSleep", 0.1, 1e-6, 123, -1, -1, +1, "ds_b0.1_lr1e-6_s123"),
    ("DS-0.1-lr2", "DeepSleep", 0.1, 1e-6, 7,   -1, -1, +1, "ds_b0.1_lr1e-6_s7"),
    ("DS-0.5-lr2", "DeepSleep", 0.5, 1e-6, 42,  -1, +1, +1, "ds_b0.5_lr1e-6_s42"),
    ("DS-0.5-lr2", "DeepSleep", 0.5, 1e-6, 123, -1, +1, +1, "ds_b0.5_lr1e-6_s123"),
    ("DS-0.5-lr2", "DeepSleep", 0.5, 1e-6, 7,   -1, +1, +1, "ds_b0.5_lr1e-6_s7"),
    ("QW-0.1-lr2", "Qwen",      0.1, 1e-6, 42,  +1, -1, +1, "qwen_b0.1_lr1e-6_s42"),
    ("QW-0.1-lr2", "Qwen",      0.1, 1e-6, 123, +1, -1, +1, "qwen_b0.1_lr1e-6_s123"),
    ("QW-0.1-lr2", "Qwen",      0.1, 1e-6, 7,   +1, -1, +1, "qwen_b0.1_lr1e-6_s7"),
    ("QW-0.5-lr2", "Qwen",      0.5, 1e-6, 42,  +1, +1, +1, "qwen_b0.5_lr1e-6_s42"),
    ("QW-0.5-lr2", "Qwen",      0.5, 1e-6, 123, +1, +1, +1, "qwen_b0.5_lr1e-6_s123"),
    ("QW-0.5-lr2", "Qwen",      0.5, 1e-6, 7,   +1, +1, +1, "qwen_b0.5_lr1e-6_s7"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_experiment(dirname):
    base = os.path.join(EXPERIMENT_DIR, dirname)
    with open(os.path.join(base, "report.json")) as f:
        report = json.load(f)
    train_log = []
    with open(os.path.join(base, "train_log.jsonl")) as f:
        for line in f:
            train_log.append(json.loads(line.strip()))
    return report, train_log


def load_all_data():
    rows = []
    train_curves = {}
    missing = []
    for group, model, beta, lr, seed, cA, cB, cC, dirname in EXPERIMENTS:
        path = os.path.join(EXPERIMENT_DIR, dirname)
        if not os.path.isdir(path):
            missing.append(dirname)
            continue
        report_path = os.path.join(path, "report.json")
        if not os.path.isfile(report_path):
            missing.append(dirname)
            continue
        report, train_log = load_experiment(dirname)
        loss_step50 = train_log[0]["loss"] if train_log else None
        final_loss = report["final_loss"]

        # Y1: Final DPO Loss
        y1 = final_loss
        # Y2: Loss Reduction Ratio
        y2 = (loss_step50 - final_loss) / loss_step50 if loss_step50 else None
        # Y3: Normalized AUC
        steps = np.array([e["step"] for e in train_log], dtype=float)
        losses = np.array([e["loss"] for e in train_log], dtype=float)
        if len(steps) > 1:
            s_norm = (steps - steps[0]) / (steps[-1] - steps[0])
            y3 = float(np.trapezoid(losses, s_norm))
        else:
            y3 = None

        rows.append({
            "group": group, "model": model, "beta": beta, "lr": lr,
            "seed": seed, "A": cA, "B": cB, "C": cC,
            "dirname": dirname,
            "final_loss": final_loss, "loss_step50": loss_step50,
            "accuracy": report["final_accuracy"],
            "total_steps": report["total_steps"],
            "total_time": report["total_time_hours"],
            "Y1_loss": y1, "Y2_reduction": y2, "Y3_auc": y3,
        })
        train_curves[dirname] = train_log

    if missing:
        print(f"\n  WARNING: {len(missing)} experiment directories missing:")
        for m in missing:
            print(f"    - {m}")
        print(f"  Loaded {len(rows)}/24 experiments. ANOVA requires all 24.\n")

    return pd.DataFrame(rows), train_curves


# ═══════════════════════════════════════════════════════════════════════════
# ANOVA Computation (2^3 Full Factorial)
# ═══════════════════════════════════════════════════════════════════════════

def sig_mark(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def compute_anova(df, col):
    """Compute 2^3 full factorial ANOVA using orthogonal contrasts.

    Model: Y = mu + aA + bB + cC + abAB + acAC + bcBC + abcABC + epsilon
    """
    y = df[col].values.astype(float)
    A = df["A"].values.astype(float)
    B = df["B"].values.astype(float)
    C = df["C"].values.astype(float)
    N = len(y)
    n_treatments = 8  # 2^3
    r = N // n_treatments

    y_bar = np.mean(y)
    SS_T = np.sum((y - y_bar) ** 2)

    # ── Orthogonal contrast SS: SS_effect = (sum(c_i * y_i))^2 / N ──
    def contrast_ss(*factor_cols):
        c = np.ones(N)
        for fc in factor_cols:
            c *= df[fc].values.astype(float)
        return (c @ y) ** 2 / N

    SS_A   = contrast_ss("A")
    SS_B   = contrast_ss("B")
    SS_C   = contrast_ss("C")
    SS_AB  = contrast_ss("A", "B")
    SS_AC  = contrast_ss("A", "C")
    SS_BC  = contrast_ss("B", "C")
    SS_ABC = contrast_ss("A", "B", "C")
    SS_E   = SS_T - SS_A - SS_B - SS_C - SS_AB - SS_AC - SS_BC - SS_ABC

    # Degrees of freedom
    df_A = df_B = df_C = df_AB = df_AC = df_BC = df_ABC = 1
    df_E = N - n_treatments
    df_T = N - 1

    # Mean squares
    MS_A   = SS_A / df_A
    MS_B   = SS_B / df_B
    MS_C   = SS_C / df_C
    MS_AB  = SS_AB / df_AB
    MS_AC  = SS_AC / df_AC
    MS_BC  = SS_BC / df_BC
    MS_ABC = SS_ABC / df_ABC
    MS_E   = SS_E / df_E

    # F and p
    def fp(MS_eff, df_eff=1):
        F = MS_eff / MS_E
        p = 1 - stats.f.cdf(F, df_eff, df_E)
        return F, p

    F_A, p_A         = fp(MS_A)
    F_B, p_B         = fp(MS_B)
    F_C, p_C         = fp(MS_C)
    F_AB, p_AB       = fp(MS_AB)
    F_AC, p_AC       = fp(MS_AC)
    F_BC, p_BC       = fp(MS_BC)
    F_ABC, p_ABC     = fp(MS_ABC)

    # Effect estimates (mean at +1 minus mean at -1)
    eff_A = np.mean(y[A == 1]) - np.mean(y[A == -1])
    eff_B = np.mean(y[B == 1]) - np.mean(y[B == -1])
    eff_C = np.mean(y[C == 1]) - np.mean(y[C == -1])

    # Fitted values and residuals
    cells = {}
    for i in range(N):
        key = (A[i], B[i], C[i])
        cells.setdefault(key, []).append(y[i])
    fitted = np.array([np.mean(cells[(A[i], B[i], C[i])]) for i in range(N)])
    residuals = y - fitted

    R2 = 1 - SS_E / SS_T
    R2_adj = 1 - (SS_E / df_E) / (SS_T / df_T)

    return {
        "src": [
            "Factor A (Architecture)", "Factor B (DPO Beta)",
            "Factor C (Learning Rate)",
            "A × B", "A × C", "B × C", "A × B × C",
            "Error", "Total",
        ],
        "df": [df_A, df_B, df_C, df_AB, df_AC, df_BC, df_ABC, df_E, df_T],
        "SS": [SS_A, SS_B, SS_C, SS_AB, SS_AC, SS_BC, SS_ABC, SS_E, SS_T],
        "MS": [MS_A, MS_B, MS_C, MS_AB, MS_AC, MS_BC, MS_ABC, MS_E, None],
        "F":  [F_A, F_B, F_C, F_AB, F_AC, F_BC, F_ABC, None, None],
        "p":  [p_A, p_B, p_C, p_AB, p_AC, p_BC, p_ABC, None, None],
        "sig": [sig_mark(p_A), sig_mark(p_B), sig_mark(p_C),
                sig_mark(p_AB), sig_mark(p_AC), sig_mark(p_BC), sig_mark(p_ABC),
                "", ""],
        "R2": R2, "R2_adj": R2_adj,
        "eff_A": eff_A, "eff_B": eff_B, "eff_C": eff_C,
        "fitted": fitted, "residuals": residuals,
    }


def print_anova(anova, name):
    print(f"\n{'=' * 100}")
    print(f"  ANOVA Table: {name}")
    print(f"{'=' * 100}")
    hdr = f"{'Source':<28} {'df':>3} {'SS':>14} {'MS':>14} {'F':>10} {'p-value':>10} {'Sig.':>5}"
    print(hdr)
    print("-" * 100)
    for i, s in enumerate(anova["src"]):
        d = anova["df"][i]
        ss = f"{anova['SS'][i]:.4e}" if anova['SS'][i] is not None else ""
        ms = f"{anova['MS'][i]:.4e}" if anova['MS'][i] is not None else ""
        f = f"{anova['F'][i]:.2f}" if anova['F'][i] is not None else ""
        p = f"{anova['p'][i]:.4f}" if anova['p'][i] is not None else ""
        print(f"{s:<28} {d:>3} {ss:>14} {ms:>14} {f:>10} {p:>10} {anova['sig'][i]:>5}")
    print("-" * 100)
    print(f"  R² = {anova['R2']:.4f}   R²(adj) = {anova['R2_adj']:.4f}")
    print(f"  Effect A = {anova['eff_A']:.4e}   Effect B = {anova['eff_B']:.4e}   Effect C = {anova['eff_C']:.4e}")
    print(f"{'=' * 100}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Publication-Quality Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")


def _format_sci(ax, axis="y"):
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


# ── Fig 0: Summary Statistics Table ────────────────────────────────────────

def plot_summary_table(df, responses):
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.axis("off")

    col_labels = ["Treatment", "n",
                  "$Y_1$: Loss\n(mean $\\pm$ SE)",
                  "$Y_2$: Reduction\n(mean $\\pm$ SE)",
                  "$Y_3$: AUC\n(mean $\\pm$ SE)"]

    rows_data = []
    for model, beta, lr, label in [
        ("DeepSleep", 0.1, 5e-7, "DS MoE, β=0.1, LR=5e-7"),
        ("DeepSleep", 0.1, 1e-6, "DS MoE, β=0.1, LR=1e-6"),
        ("DeepSleep", 0.5, 5e-7, "DS MoE, β=0.5, LR=5e-7"),
        ("DeepSleep", 0.5, 1e-6, "DS MoE, β=0.5, LR=1e-6"),
        ("Qwen",      0.1, 5e-7, "Qwen Dense, β=0.1, LR=5e-7"),
        ("Qwen",      0.1, 1e-6, "Qwen Dense, β=0.1, LR=1e-6"),
        ("Qwen",      0.5, 5e-7, "Qwen Dense, β=0.5, LR=5e-7"),
        ("Qwen",      0.5, 1e-6, "Qwen Dense, β=0.5, LR=1e-6"),
    ]:
        sub = df[(df["model"] == model) & (df["beta"] == beta) & (df["lr"] == lr)]
        if len(sub) == 0:
            rows_data.append([label, "0", "—", "—", "—"])
            continue
        rows_data.append([
            label, str(len(sub)),
            f"{sub['Y1_loss'].mean():.1e} $\\pm$ {sub['Y1_loss'].sem():.1e}",
            f"{sub['Y2_reduction'].mean():.3f} $\\pm$ {sub['Y2_reduction'].sem():.3f}",
            f"{sub['Y3_auc'].mean():.3f} $\\pm$ {sub['Y3_auc'].sem():.3f}",
        ])

    tbl = ax.table(cellText=rows_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(PAL["blue"])
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows_data) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_edgecolor("0.85")
            if i % 2 == 0:
                tbl[i, j].set_facecolor("#F0F4FF")

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig0_summary_table.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 1: Training Curves (2×4 grid) ─────────────────────────────────────

def plot_training_curves(df, curves):
    """Fig 1: 8-panel training loss curves."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))

    configs = [
        ("DeepSleep MoE, β=0.1, LR=5e-7",  ["ds_b0.1_s42", "ds_b0.1_s123", "ds_b0.1_s7"], PAL["light_blue"]),
        ("DeepSleep MoE, β=0.5, LR=5e-7",  ["ds_b0.5_s42", "ds_b0.5_s123", "ds_b0.5_s7"], PAL["blue"]),
        ("Qwen Dense, β=0.1, LR=5e-7",     ["qwen_b0.1_s42", "qwen_b0.1_s123", "qwen_b0.1_s7"], PAL["light_red"]),
        ("Qwen Dense, β=0.5, LR=5e-7",     ["qwen_b0.5_s42", "qwen_b0.5_s123", "qwen_b0.5_s7"], PAL["red"]),
        ("DeepSleep MoE, β=0.1, LR=1e-6",  ["ds_b0.1_lr1e-6_s42", "ds_b0.1_lr1e-6_s123", "ds_b0.1_lr1e-6_s7"], "#66C2A5"),
        ("DeepSleep MoE, β=0.5, LR=1e-6",  ["ds_b0.5_lr1e-6_s42", "ds_b0.5_lr1e-6_s123", "ds_b0.5_lr1e-6_s7"], PAL["teal"]),
        ("Qwen Dense, β=0.1, LR=1e-6",     ["qwen_b0.1_lr1e-6_s42", "qwen_b0.1_lr1e-6_s123", "qwen_b0.1_lr1e-6_s7"], PAL["yellow"]),
        ("Qwen Dense, β=0.5, LR=1e-6",     ["qwen_b0.5_lr1e-6_s42", "qwen_b0.5_lr1e-6_s123", "qwen_b0.5_lr1e-6_s7"], PAL["purple"]),
    ]
    labels = list("abcdefgh")
    seeds = [42, 123, 7]
    lstyles = ["-", "--", ":"]

    for idx, (title, dirs, color) in enumerate(configs):
        ax = axes.flat[idx]
        for j, d in enumerate(dirs):
            if d not in curves:
                continue
            log = curves[d]
            s = [e["step"] for e in log]
            l = [e["loss"] for e in log]
            ax.plot(s, l, color=color, ls=lstyles[j], lw=1.3, alpha=0.85,
                    label=f"seed = {seeds[j]}")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("DPO Loss")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True)
        _format_sci(ax, "y")
        _panel_label(ax, labels[idx])

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    path = os.path.join(FIGURE_DIR, "fig1_training_curves.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)

    # ── Fig 1 supplement: overlay all 24 runs ─────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for _, row in df.iterrows():
        g = row["group"]
        c = GROUP_COLORS.get(g, PAL["grey"])
        ls = "--" if "lr2" in g else "-"
        log = curves.get(row["dirname"], [])
        if not log:
            continue
        s = [e["step"] for e in log]
        l = [e["loss"] for e in log]
        ax.plot(s, l, color=c, ls=ls, lw=1.0, alpha=0.5)

    handles = [
        Line2D([], [], color=PAL["light_blue"], lw=2, label="DS β=0.1 LR=5e-7"),
        Line2D([], [], color="#66C2A5", lw=2, ls="--", label="DS β=0.1 LR=1e-6"),
        Line2D([], [], color=PAL["blue"], lw=2, label="DS β=0.5 LR=5e-7"),
        Line2D([], [], color=PAL["teal"], lw=2, ls="--", label="DS β=0.5 LR=1e-6"),
        Line2D([], [], color=PAL["light_red"], lw=2, label="QW β=0.1 LR=5e-7"),
        Line2D([], [], color=PAL["yellow"], lw=2, ls="--", label="QW β=0.1 LR=1e-6"),
        Line2D([], [], color=PAL["red"], lw=2, label="QW β=0.5 LR=5e-7"),
        Line2D([], [], color=PAL["purple"], lw=2, ls="--", label="QW β=0.5 LR=1e-6"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("DPO Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    _format_sci(ax, "y")
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig1s_overlay.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 2: Main Effects Plot ──────────────────────────────────────────────

def plot_main_effects(df, anova_results, responses):
    """Fig 2: Main effects plot for A, B, C per response variable."""
    n = len(responses)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 1.5, 3.2))
    if n == 1:
        axes = [axes]

    for i, (col, label) in enumerate(responses):
        ax = axes[i]
        a = anova_results[col]

        means_A = df.groupby("A")[col].mean()
        means_B = df.groupby("B")[col].mean()
        means_C = df.groupby("C")[col].mean()

        ax.plot([-1, 1], [means_A[-1], means_A[1]], "o-",
                color=PAL["blue"], lw=2, ms=7, zorder=5,
                label=f"A: Arch (p={a['p'][0]:.3f} {a['sig'][0]})")
        ax.plot([-1, 1], [means_B[-1], means_B[1]], "s--",
                color=PAL["red"], lw=2, ms=7, zorder=5,
                label=f"B: Beta (p={a['p'][1]:.3f} {a['sig'][1]})")
        ax.plot([-1, 1], [means_C[-1], means_C[1]], "^-.",
                color=PAL["teal"], lw=2, ms=7, zorder=5,
                label=f"C: LR (p={a['p'][2]:.3f} {a['sig'][2]})")

        ax.axhline(df[col].mean(), color=PAL["grey"], ls=":", lw=0.8)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(["$-1$", "$+1$"])
        ax.set_xlabel("Coded Level")
        ax.set_ylabel(label)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True)
        _panel_label(ax, "abc"[i], x=-0.22, y=1.05)

    fig.tight_layout(w_pad=1.8)
    path = os.path.join(FIGURE_DIR, "fig2_main_effects.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 3: Two-Way Interaction Plots ───────────────────────────────────────

def plot_interaction(df, responses):
    """Fig 3: A×B, A×C, B×C interaction plots per response."""
    n = len(responses)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (col, label) in enumerate(responses):
        for j, (fx, fy, title, ax) in enumerate([
            ("model", "beta", "A × B: Architecture × Beta", axes[i, 0]),
            ("model", "lr",   "A × C: Architecture × LR",  axes[i, 1]),
            ("beta",  "lr",   "B × C: Beta × LR",           axes[i, 2]),
        ]):
            means = df.groupby([fx, fy])[col].mean()
            sems = df.groupby([fx, fy])[col].sem()

            x_vals = sorted(df[fy].unique())
            for k, (level, color, mk) in enumerate([
                (df[fx].unique()[0], PAL["blue"], "o"),
                (df[fx].unique()[-1], PAL["red"], "s"),
            ]):
                y = [means[(level, xv)] for xv in x_vals if (level, xv) in means.index]
                e = [sems[(level, xv)] for xv in x_vals if (level, xv) in sems.index]
                valid_x = [xv for xv in x_vals if (level, xv) in means.index]
                if len(y) == len(valid_x):
                    ax.errorbar(valid_x, y, yerr=e, marker=mk, color=color,
                                lw=2, ms=7, capsize=4, capthick=1.2,
                                label=str(level), zorder=5)

            ax.set_xlabel(fy.capitalize())
            ax.set_ylabel(label if j == 0 else "")
            ax.set_title(title, fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True)

        _panel_label(axes[i, 0], "abc"[i] * 1, x=-0.22, y=1.05)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    path = os.path.join(FIGURE_DIR, "fig3_interaction.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 4: Three-Way Interaction (Conditional A×B at C=-1/+1) ─────────────

def plot_threeway_interaction(df, responses):
    """Fig 4: A×B×C interaction shown as conditional A×B plots at each C level."""
    n = len(responses)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (col, label) in enumerate(responses):
        for j, (lr_val, lr_label, ax) in enumerate([
            (5e-7, "C = −1 (LR = 5e-7)", axes[i, 0]),
            (1e-6, "C = +1 (LR = 1e-6)", axes[i, 1]),
        ]):
            sub = df[df["lr"] == lr_val]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue
            means = sub.groupby(["model", "beta"])[col].mean()
            sems = sub.groupby(["model", "beta"])[col].sem()

            for model, color, mk in [
                ("DeepSleep", PAL["blue"], "o"),
                ("Qwen", PAL["red"], "s"),
            ]:
                y = [means[(model, b)] for b in [0.1, 0.5] if (model, b) in means.index]
                e = [sems[(model, b)] for b in [0.1, 0.5] if (model, b) in sems.index]
                valid_b = [b for b in [0.1, 0.5] if (model, b) in means.index]
                if len(y) == len(valid_b):
                    ax.errorbar(valid_b, y, yerr=e, marker=mk, color=color,
                                lw=2, ms=7, capsize=4, capthick=1.2,
                                label=model, zorder=5)

            ax.set_title(lr_label, fontsize=8.5)
            ax.set_xlabel("DPO β")
            ax.set_ylabel(label if j == 0 else "")
            ax.set_xticks([0.1, 0.5])
            ax.legend(fontsize=7.5)
            ax.grid(True)

        _panel_label(axes[i, 0], "abc"[i], x=-0.22, y=1.05)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    path = os.path.join(FIGURE_DIR, "fig3b_threeway.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 5: Residual Diagnostics ────────────────────────────────────────────

def plot_residuals(df, anova_results, responses):
    """Fig 5: QQ plot + Residual vs Fitted for each response."""
    for col, label in responses:
        a = anova_results.get(col)
        if a is None:
            continue
        res = a["residuals"]
        fit = a["fitted"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.6))

        # (a) Normal probability plot
        (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
        ax1.scatter(osm, osr, color=PAL["blue"], s=20, zorder=5,
                    edgecolors="white", lw=0.3)
        ax1.plot(osm, slope * np.array(osm) + intercept, color=PAL["red"], lw=1.2)
        sw_stat, sw_p = stats.shapiro(res)
        ax1.text(0.05, 0.92, f"Shapiro-Wilk: W = {sw_stat:.3f}, p = {sw_p:.3f}",
                 transform=ax1.transAxes, fontsize=6.5,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PAL["grey"], alpha=0.8))
        ax1.set_xlabel("Theoretical Quantiles")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Normal Q-Q Plot")
        ax1.grid(True)
        _panel_label(ax1, "a", x=-0.20, y=1.08)

        # (b) Residual vs Fitted
        ax2.scatter(fit, res, color=PAL["blue"], s=20, zorder=5,
                    edgecolors="white", lw=0.3)
        ax2.axhline(0, color=PAL["grey"], ls="--", lw=0.8)
        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs. Fitted")
        ax2.grid(True)
        _format_sci(ax2, "x")
        _panel_label(ax2, "b", x=-0.20, y=1.08)

        fig.tight_layout(w_pad=2.0)
        tag = col.replace(" ", "_")
        path = os.path.join(FIGURE_DIR, f"fig4_residuals_{tag}.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
        plt.close(fig)


# ── Fig 6: Pareto Chart of Effects ────────────────────────────────────────

def plot_pareto(anova_results, responses):
    """Fig 6: Pareto chart of F-values for all 7 effects."""
    n = len(responses)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 2, 3.5))
    if n == 1:
        axes = [axes]

    df_error = 16  # df_Error for 2^3 × 3
    F_crit = stats.f.ppf(0.95, 1, df_error)
    eff_names = ["A (Arch)", "B (Beta)", "C (LR)",
                 "A × B", "A × C", "B × C", "A × B × C"]

    for i, (col, label) in enumerate(responses):
        ax = axes[i]
        a = anova_results[col]
        F_vals = a["F"][:7]
        p_vals = a["p"][:7]

        # Sort descending by |F|
        idx = np.argsort(np.abs(F_vals))[::-1]
        names = [eff_names[j] for j in idx]
        fs = [F_vals[j] for j in idx]
        ps = [p_vals[j] for j in idx]

        bar_colors = [PAL["blue"] if p < 0.05 else PAL["grey"] for p in ps]
        ax.barh(range(7), fs, color=bar_colors, edgecolor="white", lw=0.8, height=0.55)
        ax.set_yticks(range(7))
        ax.set_yticklabels(names, fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(F_crit, color=PAL["red"], ls="--", lw=1.2,
                   label=f"$F_{{0.05}}$(1,{df_error}) = {F_crit:.1f}")
        ax.set_xlabel("F-value")
        ax.legend(fontsize=6.5, loc="lower right")

        for j, (fv, pv) in enumerate(zip(fs, ps)):
            s = sig_mark(pv)
            ax.text(fv + 0.3, j, f"p={pv:.3f} {s}", va="center", fontsize=6)

        _panel_label(ax, "abc"[i], x=-0.38, y=1.05)

    fig.tight_layout(w_pad=2.0)
    path = os.path.join(FIGURE_DIR, "fig6_pareto.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 7: Box + Strip Plots ──────────────────────────────────────────────

def plot_boxplots(df, responses):
    """Fig 7: Box + strip plots for 8 treatment groups."""
    n = len(responses)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n + 2, 3.5))
    if n == 1:
        axes = [axes]

    pal = [GROUP_COLORS[g] for g in GROUP_ORDER]

    for i, (col, label) in enumerate(responses):
        ax = axes[i]
        bp = sns.boxplot(data=df, x="group", y=col, order=GROUP_ORDER,
                         palette=pal, ax=ax, width=0.55, linewidth=0.8,
                         fliersize=0, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df, x="group", y=col, order=GROUP_ORDER,
                      color=PAL["black"], size=3.5, alpha=0.6, ax=ax, jitter=0.15)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=6)
        ax.grid(True, axis="y")
        _panel_label(ax, "abc"[i], x=-0.25, y=1.05)

    fig.tight_layout(w_pad=1.5)
    path = os.path.join(FIGURE_DIR, "fig5_boxplots.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 8: Normalized Convergence Curves ──────────────────────────────────

def plot_convergence(df, curves):
    """Fig 8: Normalized convergence curves with confidence bands for 8 groups."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    groups = {}
    for _, row in df.iterrows():
        g = row["group"]
        groups.setdefault(g, []).append(row["dirname"])

    x_grid = np.linspace(0, 100, 200)
    for gname, dirnames in sorted(groups.items()):
        color = GROUP_COLORS.get(gname, PAL["grey"])
        ls = "--" if "lr2" in gname else "-"
        interp_all = []
        for d in dirnames:
            if d not in curves:
                continue
            log = curves[d]
            s = np.array([e["step"] for e in log], dtype=float)
            l = np.array([e["loss"] for e in log], dtype=float)
            if s[-1] > s[0]:
                s_norm = (s - s[0]) / (s[-1] - s[0]) * 100
            else:
                s_norm = s
            interp_all.append(np.interp(x_grid, s_norm, l))
        if not interp_all:
            continue
        interp_all = np.array(interp_all)
        mean = interp_all.mean(axis=0)
        std = interp_all.std(axis=0)

        ax.plot(x_grid, mean, color=color, ls=ls, lw=2, label=gname, zorder=5)
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("DPO Loss")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 100)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig8_convergence.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(df, anova_results, responses):
    r = []
    r.append("# 2^3 Full Factorial Experiment Analysis Report\n")
    r.append("> Course: Scientific Experiment Analysis | Experiment: DPO Preference Alignment\n")
    r.append("> Design: 2^3 full factorial (Architecture × DPO Beta × Learning Rate), 3 replications\n")
    r.append("---\n")

    # 1. Raw data
    r.append("## 1. Raw Experimental Data\n")
    r.append("| Run | Model | Beta | LR | Seed | Y1: Final Loss | Y2: Reduction | Y3: AUC |")
    r.append("|-----|-------|------|-----|------|----------------|---------------|---------|")
    for i, row in df.iterrows():
        r.append(f"| {i+1} | {row['model']} | {row['beta']} | {row['lr']:.0e} | {row['seed']} | "
                 f"{row['Y1_loss']:.4e} | {row['Y2_reduction']:.4f} | {row['Y3_auc']:.4f} |")
    r.append("")

    # 2. Descriptive stats
    r.append("## 2. Descriptive Statistics\n")
    r.append("| Treatment | n | Y1 (mean +/- SE) | Y2 (mean +/- SE) | Y3 (mean +/- SE) |")
    r.append("|-----------|---|------------------|------------------|------------------|")
    for model, beta, lr in [
        ("DeepSleep", 0.1, 5e-7), ("DeepSleep", 0.1, 1e-6),
        ("DeepSleep", 0.5, 5e-7), ("DeepSleep", 0.5, 1e-6),
        ("Qwen",      0.1, 5e-7), ("Qwen",      0.1, 1e-6),
        ("Qwen",      0.5, 5e-7), ("Qwen",      0.5, 1e-6),
    ]:
        s = df[(df["model"] == model) & (df["beta"] == beta) & (df["lr"] == lr)]
        if len(s) == 0:
            r.append(f"| {model} b={beta} lr={lr:.0e} | 0 | — | — | — |")
            continue
        r.append(f"| {model} b={beta} lr={lr:.0e} | {len(s)} | "
                 f"{s['Y1_loss'].mean():.2e} +/- {s['Y1_loss'].sem():.2e} | "
                 f"{s['Y2_reduction'].mean():.4f} +/- {s['Y2_reduction'].sem():.4f} | "
                 f"{s['Y3_auc'].mean():.4f} +/- {s['Y3_auc'].sem():.4f} |")
    r.append("")

    # 3. ANOVA
    r.append("## 3. ANOVA Tables\n")
    for col, label in responses:
        a = anova_results.get(col)
        if a is None:
            continue
        r.append(f"### {label}\n")
        r.append("| Source | df | SS | MS | F | p-value | Sig. |")
        r.append("|--------|----|----|----|---|---------|------|")
        for i in range(9):
            ss = f"{a['SS'][i]:.4e}" if a['SS'][i] is not None else ""
            ms = f"{a['MS'][i]:.4e}" if a['MS'][i] is not None else ""
            fv = f"{a['F'][i]:.2f}" if a['F'][i] is not None else ""
            pv = f"{a['p'][i]:.4f}" if a['p'][i] is not None else ""
            r.append(f"| {a['src'][i]} | {a['df'][i]} | {ss} | {ms} | {fv} | {pv} | {a['sig'][i]} |")
        r.append(f"\n**R-squared = {a['R2']:.4f}, R-squared(adj) = {a['R2_adj']:.4f}**\n")
        r.append(f"- Effect A (Architecture): {a['eff_A']:.4e}")
        r.append(f"- Effect B (DPO Beta): {a['eff_B']:.4e}")
        r.append(f"- Effect C (Learning Rate): {a['eff_C']:.4e}\n")

    # 4. Residual diagnostics
    r.append("## 4. Residual Diagnostics\n")
    r.append("### ANOVA Assumption Validation\n")
    r.append("| Response | Shapiro-Wilk W | p-value | Normality | Interpretation |")
    r.append("|----------|---------------|---------|-----------|----------------|")
    all_resp = list(responses)
    if "Y1_log_loss" in [c for c, _ in responses]:
        pass  # already included
    # Always add log-transformed Y1
    all_resp_log = list(responses)
    for col, label in all_resp_log:
        if col not in anova_results:
            continue
        res = anova_results[col]["residuals"]
        sw_stat, sw_p = stats.shapiro(res)
        pf = "PASS" if sw_p > 0.05 else "FAIL"
        note = "Residuals approximately normal" if pf == "PASS" else "Non-normal; use log transform"
        r.append(f"| {label} | {sw_stat:.4f} | {sw_p:.4f} | {pf} | {note} |")
    r.append("")

    # 5. Discussion
    r.append("## 5. Results and Discussion\n")
    r.append("### 5.1 ANOVA Significance Summary\n")

    r.append("| Response | A (Arch.) | B (Beta) | C (LR) | A×B | A×C | B×C | A×B×C | R²(adj) |")
    r.append("|----------|-----------|----------|--------|-----|-----|-----|-------|---------|")
    for col, label in all_resp_log:
        a = anova_results.get(col)
        if a is None:
            continue
        def fmt_effect(idx):
            if a["p"][idx] is None: return "—"
            if a["p"][idx] < 0.001: return f"*** (F={a['F'][idx]:.1f})"
            if a["p"][idx] < 0.01:  return f"** (F={a['F'][idx]:.1f})"
            if a["p"][idx] < 0.05:  return f"* (F={a['F'][idx]:.1f})"
            return f"n.s. (F={a['F'][idx]:.1f})"
        r.append(f"| {label} | {fmt_effect(0)} | {fmt_effect(1)} | {fmt_effect(2)} | "
                 f"{fmt_effect(3)} | {fmt_effect(4)} | {fmt_effect(5)} | {fmt_effect(6)} | {a['R2_adj']:.3f} |")
    r.append("")
    r.append("> Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\n")

    r.append("### 5.2 Key Findings\n")
    r.append("1. **Factor C (Learning Rate) effect**: The higher learning rate (1e-6 vs 5e-7) is expected "
             "to accelerate DPO convergence but may introduce training instability, particularly for "
             "MoE architectures with sparse gradient signals.\n")
    r.append("2. **A × C interaction**: If significant, this indicates that the optimal learning rate "
             "depends on model architecture (MoE vs Dense). MoE models, with their sparse gradient "
             "flow through top-k experts, may require different LR settings than Dense models.\n")
    r.append("3. **B × C interaction**: If significant, the joint effect of Beta and LR on alignment "
             "is not simply additive — high Beta and high LR together may cause over-alignment or "
             "training instability.\n")
    r.append("4. **A × B × C three-way interaction**: The most complex effect. If significant, the "
             "optimal (Beta, LR) combination differs by architecture, requiring architecture-specific "
             "hyperparameter tuning.\n")

    r.append("### 5.3 Practical Implications\n")
    r.append("1. **Architecture-specific tuning**: If A × C is significant, MoE and Dense models "
             "should use different learning rates for DPO alignment.\n")
    r.append("2. **Joint hyperparameter optimization**: If B × C is significant, Beta and LR should "
             "be tuned jointly rather than independently.\n")
    r.append("3. **Parameter efficiency**: DeepSleep MoE's competitive performance with 7.7× fewer "
             "active parameters reinforces the value of sparse architectures for domain-specific LLMs.\n")

    r.append("### 5.4 Recommendations for Further Experiments\n")
    r.append("1. **Response surface methodology (RSM)**: Extend to continuous factors via central "
             "composite design (CCD) to locate the global optimum.\n")
    r.append("2. **Generation quality evaluation**: Use GPT-4o to score model outputs on multiple "
             "dimensions as an additional response variable.\n")
    r.append("3. **Benchmark evaluation**: PubMedQA, MedQA, ARC-Easy, PIQA, OpenBookQA.\n")
    r.append("4. **Data scaling study**: Investigate DPO dataset size effect (500/1000/2000/5000 pairs).\n")

    return "\n".join(r)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  2^3 Full Factorial ANOVA Analysis")
    print("  DPO Preference Alignment Experiments")
    print("  Factors: Architecture (A) × DPO Beta (B) × Learning Rate (C)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading experiment data...")
    df, curves = load_all_data()
    print(f"  {len(df)}/24 experiments loaded.")

    if len(df) < 24:
        print(f"\n  WARNING: Only {len(df)}/24 experiments found.")
        print("  ANOVA results will be incomplete. Run all 24 experiments first.\n")

    responses = [
        ("Y1_loss",      "$Y_1$: DPO Final Loss"),
        ("Y2_reduction", "$Y_2$: Loss Reduction Ratio"),
        ("Y3_auc",       "$Y_3$: Normalized AUC"),
    ]

    # Add log-transformed Y1 for normality correction
    df["Y1_log_loss"] = np.log10(df["Y1_loss"].clip(lower=1e-10))
    responses_log = [
        ("Y1_log_loss",  "$Y_1'$: $\\log_{10}$(Loss)"),
    ]

    all_responses = responses + responses_log

    # 2. ANOVA
    print("\n[2/7] Computing ANOVA...")
    anova_results = {}
    for col, label in all_responses:
        if col not in df.columns or df[col].isna().any():
            print(f"  Skipping {label} (missing data)")
            continue
        anova_results[col] = compute_anova(df, col)
        print_anova(anova_results[col], label)

    # Shapiro-Wilk on log-transformed Y1
    if "Y1_log_loss" in anova_results:
        sw_stat, sw_p = stats.shapiro(anova_results["Y1_log_loss"]["residuals"])
        print(f"  Shapiro-Wilk (log Y1): W = {sw_stat:.4f}, p = {sw_p:.4f}  "
              f"[{'PASS' if sw_p > 0.05 else 'FAIL'}]")

    # 3. Plots
    print("\n[3/7] Generating figures...")
    plot_summary_table(df, responses)
    if curves:
        plot_training_curves(df, curves)
        plot_convergence(df, curves)
    plot_main_effects(df, anova_results, responses)
    plot_interaction(df, responses)
    plot_threeway_interaction(df, responses)
    plot_residuals(df, anova_results, responses)
    plot_pareto(anova_results, responses)
    plot_boxplots(df, responses)

    # 4. Report
    print("\n[4/7] Writing report...")
    report = generate_report(df, anova_results, responses)
    rpath = os.path.join(FIGURE_DIR, "..", "analysis_report.md")
    with open(rpath, "w") as f:
        f.write(report)
    print(f"  Saved: {rpath}")

    # 5. CSV
    print("\n[5/7] Saving CSV...")
    csv_path = os.path.join(FIGURE_DIR, "experiment_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 6. LaTeX tables
    print("\n[6/7] LaTeX ANOVA tables...")
    for col, label in responses:
        a = anova_results.get(col)
        if a is None:
            continue
        print(f"\n% --- {label} ---")
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(r"\caption{" + label + " ANOVA Table (2$^3$ Full Factorial)}")
        print(r"\begin{tabular}{lcccccc}")
        print(r"\hline")
        print(r"Source & df & SS & MS & $F$ & $p$-value \\")
        print(r"\hline")
        for i in range(9):
            ss = f"{a['SS'][i]:.4e}" if a['SS'][i] is not None else ""
            ms = f"{a['MS'][i]:.4e}" if a['MS'][i] is not None else ""
            fv = f"{a['F'][i]:.2f}" if a['F'][i] is not None else ""
            pv = f"{a['p'][i]:.4f}" if a['p'][i] is not None else ""
            print(f"{a['src'][i]} & {a['df'][i]} & {ss} & {ms} & {fv} & {pv} {a['sig'][i]} \\\\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(f"\\label{{tab:anova_{col}}}")
        print(r"\end{table}")

    print("\n[7/7] Done!")
    print(f"\n  Figures: {FIGURE_DIR}/")
    print(f"  Report:  {rpath}")
    print(f"  CSV:     {csv_path}")


if __name__ == "__main__":
    main()
