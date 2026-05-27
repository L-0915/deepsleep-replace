#!/usr/bin/env python3
"""
Pipeline Waterfall Figure: Full training lifecycle visualization.
Pretrain -> CPT -> SFT -> DPO for DeepSleep MoE
SFT -> DPO for Qwen Dense
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "mathtext.default": "regular",
})

FIGURE_DIR = "/root/dslm/deepsleep/docs/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

PAL = {
    "pretrain": "#0072B2",
    "cpt": "#56B4E9",
    "sft_ds": "#009E73",
    "sft_qw": "#D55E00",
    "dpo": "#CC79A7",
    "grey": "#999999",
}


def load_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "total_flos" in e:
                continue
            entries.append(e)
    return entries


def _panel_label(ax, label, x=-0.14, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")


def main():
    # ── Load all data ──
    base = "/root/dslm/deepsleep/out"
    pretrain = load_log(f"{base}/pretrain/train_log.jsonl")
    cpt = load_log(f"{base}/cpt/train_log.jsonl")
    sft_ds = load_log(f"{base}/sft/train_log.jsonl")
    sft_qw = load_log(f"{base}/sft_qwen/train_log.jsonl")

    dpo_exp = "/root/blockdata/dpo_exp"
    dpo_ds = load_log(f"{dpo_exp}/ds_b0.1_s42/train_log.jsonl")
    dpo_qw = load_log(f"{dpo_exp}/qwen_b0.1_s42/train_log.jsonl")

    # ── Build figure ──
    fig = plt.figure(figsize=(7.2, 6.5))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                          left=0.09, right=0.97, top=0.95, bottom=0.07)

    # ── Panel A: DeepSleep Full Pipeline ──
    ax_a = fig.add_subplot(gs[0, :])

    stages = [
        ("Pretrain", pretrain, PAL["pretrain"], 0),
        ("CPT", cpt, PAL["cpt"], pretrain[-1]["step"]),
        ("SFT", sft_ds, PAL["sft_ds"], pretrain[-1]["step"] + cpt[-1]["step"]),
        ("DPO", dpo_ds, PAL["dpo"],
         pretrain[-1]["step"] + cpt[-1]["step"] + sft_ds[-1]["step"]),
    ]

    boundaries = [0]
    for name, log, color, offset in stages:
        steps = [e["step"] + offset for e in log]
        losses = [e["loss"] for e in log]
        ax_a.plot(steps, losses, color=color, lw=1.3, alpha=0.85, label=name)
        boundaries.append(steps[-1])

        mid_x = (steps[0] + steps[-1]) / 2
        final_loss = losses[-1]
        ax_a.annotate(f"{final_loss:.2f}",
                      xy=(steps[-1], final_loss),
                      xytext=(5, 5), textcoords="offset points",
                      fontsize=6.5, color=color)

    for b in boundaries[1:-1]:
        ax_a.axvline(b, color=PAL["grey"], ls=":", lw=0.7, alpha=0.6)

    ax_a.set_xlabel("Global Training Step")
    ax_a.set_ylabel("Training Loss")
    ax_a.legend(loc="upper right", fontsize=7, ncol=4)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim(bottom=0)
    _panel_label(ax_a, "a")

    # Add stage labels at top
    for i, (name, _, color, _) in enumerate(stages):
        x_start = boundaries[i]
        x_end = boundaries[i + 1]
        x_mid = (x_start + x_end) / 2
        y_top = ax_a.get_ylim()[1]
        ax_a.text(x_mid, y_top * 0.95, name, ha="center", va="top",
                  fontsize=7.5, fontweight="bold", color=color,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white",
                            ec=color, alpha=0.85, lw=0.8))

    # ── Panel B: Qwen Pipeline ──
    ax_b = fig.add_subplot(gs[1, 0])

    qw_stages = [
        ("SFT", sft_qw, PAL["sft_qw"], 0),
        ("DPO", dpo_qw, PAL["dpo"], sft_qw[-1]["step"]),
    ]
    qw_bounds = [0]
    for name, log, color, offset in qw_stages:
        steps = [e["step"] + offset for e in log]
        losses = [e["loss"] for e in log]
        ax_b.plot(steps, losses, color=color, lw=1.3, alpha=0.85, label=name)
        qw_bounds.append(steps[-1])
        ax_b.annotate(f"{losses[-1]:.2f}",
                      xy=(steps[-1], losses[-1]),
                      xytext=(5, 5), textcoords="offset points",
                      fontsize=6.5, color=color)

    for b in qw_bounds[1:-1]:
        ax_b.axvline(b, color=PAL["grey"], ls=":", lw=0.7, alpha=0.6)

    ax_b.set_xlabel("Training Step")
    ax_b.set_ylabel("Training Loss")
    ax_b.set_title("Qwen Dense Pipeline")
    ax_b.legend(fontsize=7)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_ylim(bottom=0)
    _panel_label(ax_b, "b")

    # ── Panel C: Waterfall bar chart ──
    ax_c = fig.add_subplot(gs[1, 1])

    ds_data = [
        ("Pretrain\n(13K steps)", pretrain[0]["loss"], pretrain[-1]["loss"], PAL["pretrain"]),
        ("CPT\n(2K steps)", cpt[0]["loss"], cpt[-1]["loss"], PAL["cpt"]),
        ("SFT\n(1.6K steps)", sft_ds[0]["loss"], sft_ds[-1]["loss"], PAL["sft_ds"]),
        ("DPO\n(492 steps)", dpo_ds[0]["loss"], dpo_ds[-1]["loss"], PAL["dpo"]),
    ]

    x = np.arange(len(ds_data))
    width = 0.35

    for i, (name, start, end, color) in enumerate(ds_data):
        reduction = start - end
        ax_c.bar(i - width / 2, reduction, width, color=color, alpha=0.85,
                 edgecolor="white", lw=0.5)
        ax_c.text(i - width / 2, reduction + 0.02,
                  f"-{reduction:.2f}", ha="center", va="bottom",
                  fontsize=6.5, color=color)

    qw_data = [
        ("SFT\n(3.8K steps)", sft_qw[0]["loss"], sft_qw[-1]["loss"], PAL["sft_qw"]),
        ("DPO\n(983 steps)", dpo_qw[0]["loss"], dpo_qw[-1]["loss"], PAL["dpo"]),
    ]

    qw_x = np.arange(len(ds_data) - len(qw_data), len(ds_data))
    for i, (name, start, end, color) in enumerate(qw_data):
        idx = len(ds_data) - len(qw_data) + i
        reduction = start - end
        ax_c.bar(idx + width / 2, reduction, width, color=color, alpha=0.65,
                 edgecolor="white", lw=0.5, hatch="//")
        ax_c.text(idx + width / 2, reduction + 0.02,
                  f"-{reduction:.2f}", ha="center", va="bottom",
                  fontsize=6.5, color=color)

    ax_c.set_xticks(range(len(ds_data)))
    labels = ["Pretrain\n(13K)", "CPT\n(2K)", "SFT\nDS/QW", "DPO\nDS/QW"]
    ax_c.set_xticklabels(labels, fontsize=7)
    ax_c.set_ylabel("Loss Reduction ($\\Delta$)")

    handles = [
        mpatches.Patch(facecolor=PAL["pretrain"], label="DeepSleep"),
        mpatches.Patch(facecolor=PAL["sft_qw"], alpha=0.65, hatch="//", label="Qwen"),
    ]
    ax_c.legend(handles=handles, fontsize=7, loc="upper right")
    ax_c.grid(True, axis="y", alpha=0.3)
    _panel_label(ax_c, "c")

    # ── Save ──
    path = os.path.join(FIGURE_DIR, "fig_pipeline_waterfall.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Print summary ──
    print("\n=== Pipeline Summary ===")
    print(f"{'Stage':<15} {'Start Loss':>12} {'End Loss':>12} {'Reduction':>12}")
    print("-" * 55)
    for name, start, end, color in ds_data:
        print(f"{'DS ' + name:<15} {start:>12.4f} {end:>12.4f} {start - end:>12.4f}")
    print()
    for name, start, end, color in qw_data:
        print(f"{'QW ' + name:<15} {start:>12.4f} {end:>12.4f} {start - end:>12.4f}")


if __name__ == "__main__":
    main()
