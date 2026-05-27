#!/usr/bin/env python3
"""
SFT Comparison Figure: DeepSleep MoE vs Qwen Dense.
Single publication-quality figure with 4 panels:
  (a) Training Loss curves
  (b) Evaluation Loss (perplexity) curves
  (c) Learning rate schedule
  (d) Perplexity bar chart at final step
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Journal style ──────────────────────────────────────────────────────────
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

FIGURE_DIR = "/root/dslm/deepsleep/docs/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Colorblind-safe palette (Wong 2011)
C_DS = "#0072B2"   # Blue - DeepSleep
C_QW = "#D55E00"   # Orange-red - Qwen
C_DS_LIGHT = "#56B4E9"
C_QW_LIGHT = "#F4A582"
C_GREY = "#999999"


def _panel_label(ax, label, x=-0.16, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")


def _format_sci(ax, axis="y"):
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def load_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def main():
    # ── Load data ──────────────────────────────────────────────────────────
    ds_train = load_log("/root/dslm/deepsleep/out/sft/train_log.jsonl")
    qw_train = load_log("/root/dslm/deepsleep/out/sft_qwen/train_log.jsonl")
    ds_eval = load_log("/root/dslm/deepsleep/out/sft/eval_log.jsonl")
    qw_eval = load_log("/root/dslm/deepsleep/out/sft_qwen/eval_log.jsonl")

    ds_report = json.load(open("/root/dslm/deepsleep/out/sft/report.json"))
    qw_report = json.load(open("/root/dslm/deepsleep/out/sft_qwen/report.json"))

    ds_steps = [e["step"] for e in ds_train]
    ds_loss = [e["loss"] for e in ds_train]
    ds_aux = [e.get("aux_loss", 0) for e in ds_train]

    qw_steps = [e["step"] for e in qw_train]
    qw_loss = [e["loss"] for e in qw_train]

    ds_eval_steps = [e["step"] for e in ds_eval]
    ds_eval_loss = [e["eval_loss"] for e in ds_eval]
    qw_eval_steps = [e["step"] for e in qw_eval]
    qw_eval_loss = [e["eval_loss"] for e in qw_eval]

    # ── Create figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 5.4))

    # GridSpec: 2 rows, 2 cols
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.40,
                          left=0.10, right=0.96, top=0.94, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, 0])  # (a) Train loss
    ax2 = fig.add_subplot(gs[0, 1])  # (b) Eval perplexity
    ax3 = fig.add_subplot(gs[1, 0])  # (c) Learning rate
    ax4 = fig.add_subplot(gs[1, 1])  # (d) Final perplexity bar chart

    # ── (a) Training Loss ─────────────────────────────────────────────────
    ax1.plot(ds_steps, ds_loss, color=C_DS, lw=1.4, alpha=0.85, label="DeepSleep MoE")
    ax1.plot(qw_steps, qw_loss, color=C_QW, lw=1.4, alpha=0.85, label="Qwen Dense")
    # Secondary y-axis for aux_loss
    ax1b = ax1.twinx()
    ax1b.plot(ds_steps, ds_aux, color=C_DS_LIGHT, lw=1.0, ls="--", alpha=0.6)
    ax1b.set_ylabel("MoE Aux Loss", fontsize=7.5, color=C_DS_LIGHT)
    ax1b.tick_params(axis="y", labelsize=7, colors=C_DS_LIGHT)
    ax1b.spines["right"].set_visible(True)
    ax1b.spines["right"].set_color(C_DS_LIGHT)
    ax1b.spines["top"].set_visible(False)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("SFT Loss")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True)
    _panel_label(ax1, "a")

    # ── (b) Eval Perplexity ───────────────────────────────────────────────
    ds_ppl = [np.exp(l) for l in ds_eval_loss]
    qw_ppl = [np.exp(l) for l in qw_eval_loss]

    ax2.plot(ds_eval_steps, ds_ppl, color=C_DS, lw=1.8, marker="o", ms=5, label="DeepSleep MoE")
    ax2.plot(qw_eval_steps, qw_ppl, color=C_QW, lw=1.8, marker="s", ms=5, label="Qwen Dense")

    # Annotate final values
    ax2.annotate(f"{ds_ppl[-1]:.1f}", xy=(ds_eval_steps[-1], ds_ppl[-1]),
                 xytext=(10, 8), textcoords="offset points", fontsize=7, color=C_DS,
                 arrowprops=dict(arrowstyle="-", color=C_DS, lw=0.6))
    ax2.annotate(f"{qw_ppl[-1]:.1f}", xy=(qw_eval_steps[-1], qw_ppl[-1]),
                 xytext=(10, -12), textcoords="offset points", fontsize=7, color=C_QW,
                 arrowprops=dict(arrowstyle="-", color=C_QW, lw=0.6))

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Evaluation Perplexity")
    ax2.legend(fontsize=7)
    ax2.grid(True)
    _panel_label(ax2, "b")

    # ── (c) Learning Rate Schedule ─────────────────────────────────────────
    ds_lr = [e["lr"] for e in ds_train]
    qw_lr = [e["lr"] for e in qw_train]

    ax3.plot(ds_steps, [l * 1e6 for l in ds_lr], color=C_DS, lw=1.4, alpha=0.85, label="DeepSleep MoE")
    ax3.plot(qw_steps, [l * 1e6 for l in qw_lr], color=C_QW, lw=1.4, alpha=0.85, label="Qwen Dense")
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Learning Rate ($\\times 10^{-6}$)")
    ax3.legend(fontsize=7)
    ax3.grid(True)
    _format_sci(ax3, "y")
    _panel_label(ax3, "c")

    # ── (d) Final Metrics Bar Chart ────────────────────────────────────────
    metrics = ["Train Loss", "Eval Loss", "Perplexity"]
    ds_vals = [ds_report["final_loss"], ds_eval_loss[-1], ds_ppl[-1]]
    qw_vals = [qw_report["final_loss"], qw_eval_loss[-1], qw_ppl[-1]]

    x = np.arange(len(metrics))
    w = 0.32

    bars1 = ax4.bar(x - w/2, ds_vals, w, color=C_DS, label="DeepSleep MoE", edgecolor="white", lw=0.5)
    bars2 = ax4.bar(x + w/2, qw_vals, w, color=C_QW, label="Qwen Dense", edgecolor="white", lw=0.5)

    # Value labels
    for bar, val in zip(bars1, ds_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=C_DS)
    for bar, val in zip(bars2, qw_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=C_QW)

    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel("Value")
    ax4.legend(fontsize=7.5)
    ax4.grid(True, axis="y")
    ax4.set_ylim(0, max(max(ds_vals), max(qw_vals)) * 1.15)
    _panel_label(ax4, "d")

    # ── Save ───────────────────────────────────────────────────────────────
    path = os.path.join(FIGURE_DIR, "fig_sft_comparison.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n=== SFT Comparison Summary ===")
    print(f"{'Metric':<20} {'DeepSleep MoE':>15} {'Qwen Dense':>15}")
    print("-" * 52)
    print(f"{'Total Steps':<20} {ds_report['total_steps']:>15} {qw_report['total_steps']:>15}")
    print(f"{'Epochs':<20} {ds_report['epochs']:>15} {qw_report['epochs']:>15}")
    print(f"{'Train Time (h)':<20} {ds_report['total_time_hours']:>15.2f} {qw_report['total_time_hours']:>15.2f}")
    print(f"{'Final Train Loss':<20} {ds_report['final_loss']:>15.4f} {qw_report['final_loss']:>15.4f}")
    print(f"{'Final Eval Loss':<20} {ds_eval_loss[-1]:>15.4f} {qw_eval_loss[-1]:>15.4f}")
    print(f"{'Final Perplexity':<20} {ds_ppl[-1]:>15.1f} {qw_ppl[-1]:>15.1f}")
    print(f"{'Params (total)':<20} {'~199M':>15} {'~494M':>15}")
    print(f"{'Params (active)':<20} {'~64.5M':>15} {'~494M':>15}")
    print(f"{'LR':<20} {'5e-6':>15} {'5e-6':>15}")
    print(f"{'Batch Size':<20} {ds_report['config']['batch_size']:>15} {qw_report['config']['batch_size']:>15}")


if __name__ == "__main__":
    main()
