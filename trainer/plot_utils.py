"""Publication-quality training curve plotting.

Style inspired by NeurIPS/ICML/ICLR figures:
  - Clean, minimal design with subtle grid
  - Professional color palette
  - EMA smoothing on noisy train loss
  - Best-checkpoint annotation
  - 300 DPI print-ready output
"""

import os
import math


def _setup_style():
    """Set publication-quality matplotlib style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        # Clean spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })
    return plt


# Professional color palette (Wong palette for colorblind safety)
COLORS = {
    "train_loss": "#0072B2",     # blue
    "eval_loss": "#D55E00",      # vermillion
    "lr": "#009E73",             # green
    "perplexity": "#CC79A7",     # pink
    "aux_loss": "#56B4E9",       # sky blue
}


def _ema(values, span=0.05):
    """Exponential moving average for smooth curves."""
    if not values:
        return values
    alpha = min(span, 2.0 / (len(values) + 1))
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _annotate_best(ax, steps, values, color):
    """Mark the best (minimum) point with a star and value label."""
    if not steps or not values:
        return
    idx = min(range(len(values)), key=lambda i: values[i])
    ax.plot(steps[idx], values[idx], "*", color=color, markersize=12, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(
        f"{values[idx]:.3f}",
        xy=(steps[idx], values[idx]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9, lw=0.6),
    )


def plot_cpt_curves(output_dir, train_log, eval_log):
    """CPT-style 2x2: train loss, eval loss, LR, perplexity."""
    plt = _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Train Loss ---
    ax = axes[0, 0]
    entries = [(e["step"], e["loss"]) for e in train_log if "loss" in e]
    if entries:
        steps, losses = zip(*entries)
        ax.plot(steps, losses, color=COLORS["train_loss"], alpha=0.15, linewidth=0.5)
        ax.plot(steps, _ema(losses), color=COLORS["train_loss"], label="Train Loss (EMA)")
        ax.legend(loc="upper right")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    # --- Eval Loss ---
    ax = axes[0, 1]
    entries = [(e["step"], e["eval_loss"]) for e in eval_log if "eval_loss" in e]
    if entries:
        steps, losses = zip(*entries)
        ax.plot(steps, losses, "o-", color=COLORS["eval_loss"], linewidth=1.5,
                markersize=4, markeredgecolor="white", markeredgewidth=0.5)
        _annotate_best(ax, steps, losses, COLORS["eval_loss"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Evaluation Loss")

    # --- Learning Rate ---
    ax = axes[1, 0]
    entries = [(e["step"], e["lr"]) for e in train_log if "lr" in e]
    if entries:
        steps, lrs = zip(*entries)
        ax.fill_between(steps, 0, lrs, color=COLORS["lr"], alpha=0.1)
        ax.plot(steps, lrs, color=COLORS["lr"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # --- Eval Perplexity ---
    ax = axes[1, 1]
    entries = [(e["step"], e["eval_perplexity"]) for e in eval_log if "eval_perplexity" in e]
    if entries:
        steps, ppls = zip(*entries)
        ax.plot(steps, ppls, "s-", color=COLORS["perplexity"], linewidth=1.5,
                markersize=4, markeredgecolor="white", markeredgewidth=0.5)
        ax.annotate(
            f"{ppls[-1]:.2f}",
            xy=(steps[-1], ppls[-1]),
            xytext=(-50, 10), textcoords="offset points",
            fontsize=10, color=COLORS["perplexity"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["perplexity"], lw=0.8),
        )
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Perplexity")
    ax.set_title("Evaluation Perplexity")

    fig.suptitle("DeepSleep CPT Training Curves", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sft_curves(output_dir, train_log, eval_log, title="DeepSleep SFT Training Curves"):
    """SFT-style 1x3: train loss, LR, eval loss."""
    plt = _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Train Loss ---
    ax = axes[0]
    entries = [(e["step"], e["loss"]) for e in train_log if "loss" in e]
    if entries:
        steps, losses = zip(*entries)
        ax.plot(steps, losses, color=COLORS["train_loss"], alpha=0.15, linewidth=0.5)
        ax.plot(steps, _ema(losses), color=COLORS["train_loss"], label="Train Loss (EMA)")
        final = losses[-1]
        ax.annotate(
            f"{final:.4f}",
            xy=(steps[-1], final),
            xytext=(-60, 15), textcoords="offset points",
            fontsize=10, color=COLORS["train_loss"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["train_loss"], lw=0.8),
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    # --- Learning Rate ---
    ax = axes[1]
    entries = [(e["step"], e["lr"]) for e in train_log if "lr" in e]
    if entries:
        steps, lrs = zip(*entries)
        ax.fill_between(steps, 0, lrs, color=COLORS["lr"], alpha=0.1)
        ax.plot(steps, lrs, color=COLORS["lr"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # --- Eval Loss ---
    ax = axes[2]
    entries = [(e["step"], e["eval_loss"]) for e in eval_log if "eval_loss" in e]
    if entries:
        steps, losses = zip(*entries)
        ax.plot(steps, losses, "o-", color=COLORS["eval_loss"], linewidth=1.5,
                markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        _annotate_best(ax, steps, losses, COLORS["eval_loss"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Evaluation Loss")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    return path
