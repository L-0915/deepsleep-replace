"""Generate radar chart from lm-evaluation-harness benchmark results.

Reads 4 model results from benchmark_results/, creates a
publication-quality 4-model radar chart.

Usage:
    python3 scripts/eval/plot_radar.py
"""

import csv
import glob
import json
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


MODELS = [
    ("DeepSleep-Base", "ds_pretrain"),
    ("DeepSleep-DPO(β=0.1)", "ds_b0.1"),
    ("DeepSleep-DPO(β=0.5)", "ds_b0.5"),
    ("MiniMind-3", "minimind3"),
    ("Medical-GPT2", "medical_gpt2"),
    ("Qwen2.5-Base", "qw_base"),
    ("Qwen2.5-DPO(β=0.1)", "qw_b0.1"),
    ("Qwen2.5-DPO(β=0.5)", "qw_b0.5"),
]

TASK_METRICS = [
    ("PubMedQA", "pubmedqa_local", "acc,none"),
    ("MedQA", "medqa_4options", "acc_norm,none"),
    ("ARC", "arc_easy", "acc_norm,none"),
    ("PIQA", "piqa", "acc_norm,none"),
    ("OpenBookQA", "openbookqa", "acc_norm,none"),
]

COLORS = ["#E53935", "#FDD835", "#43A047", "#8E24AA", "#1E88E5", "#B5B5B5", "#EC407A", "#734D3F"]
LINESTYLES = ["-", "--", "-", "--", "-", "--", "-", "--"]
MARKERS = ["o"] * 8


def load_results(results_dir: str) -> dict:
    results = {}
    for name, prefix in MODELS:
        matches = sorted(
            [f for f in glob.glob(os.path.join(results_dir, f"{prefix}*.json")) if "_obqa_" not in f],
            reverse=True,
        )
        if not matches:
            print(f"Warning: no results for {name} (prefix: {prefix})")
            continue
        with open(matches[0]) as f:
            data = json.load(f)
        results[name] = data.get("results", {})
    return results


def extract_metrics(results: dict) -> tuple[list[str], dict[str, list[float]]]:
    labels = []
    model_scores = {}

    for label, task_key, metric_key in TASK_METRICS:
        found_for_label = False
        for model_name, task_results in results.items():
            for task_name, metrics in task_results.items():
                if task_key in task_name and metric_key in metrics:
                    val = metrics[metric_key]
                    if isinstance(val, (int, float)):
                        model_scores.setdefault(model_name, []).append(float(val) * 100)
                        found_for_label = True
                        break
        if found_for_label:
            labels.append(label)

    n = len(labels)
    for model_name in model_scores:
        while len(model_scores[model_name]) < n:
            model_scores[model_name].append(0.0)

    return labels, model_scores


def plot_radar(labels: list[str], model_scores: dict[str, list[float]], output_path: str):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    for idx, (model_name, scores) in enumerate(model_scores.items()):
        values = scores + scores[:1]
        color = COLORS[idx % len(COLORS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        mk = MARKERS[idx % len(MARKERS)]
        ax.plot(angles, values, linestyle=ls, marker=mk, linewidth=2.0,
                label=model_name, color=color, markersize=5)
        ax.fill(angles, values, alpha=0.04, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels,
        fontsize=12,
    )
    for i, label in enumerate(ax.get_xticklabels()):
        angle = angles[i]
        deg = np.degrees(angle) % 360
        if 60 < deg < 120:
            label.set_va("bottom")
        elif 240 < deg < 300:
            label.set_va("top")
    ax.tick_params(axis="x", pad=15)
    ax.set_ylim(0, 70)
    ax.set_rticks([10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.spines["polar"].set_linewidth(0.5)
    ax.spines["polar"].set_color("gray")

    ax.set_title("Benchmark: DeepSleep vs Qwen",
                 fontsize=8, pad=30)
    legend_handles = []
    for idx, model_name in enumerate(model_scores.keys()):
        kwargs = dict(color=COLORS[idx], linestyle=LINESTYLES[idx],
                      marker="o", markersize=5, linewidth=2, label=model_name)
        if LINESTYLES[idx] == "--":
            kwargs["dashes"] = (5, 2)
        h = mlines.Line2D([], [], **kwargs)
        legend_handles.append(h)
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.35, 1.12),
              fontsize=10, frameon=True, edgecolor="gray", fancybox=False,
              )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Radar chart saved: {output_path}")
    plt.close()


def main():
    results_dir = "data/eval/benchmark_results"
    output_path = "docs/figures/fig_benchmark_radar.png"

    results = load_results(results_dir)
    if not results:
        print("No results found. Run run_benchmark.sh first.")
        sys.exit(1)

    labels, model_scores = extract_metrics(results)
    if not labels:
        print("No metrics extracted. Check result file format.")
        sys.exit(1)

    print(f"Metrics: {labels}")
    for name, scores in model_scores.items():
        print(f"  {name}: {[f'{s:.4f}' for s in scores]}")

    # Save full CSV table with all metrics
    csv_path = os.path.join(os.path.dirname(output_path), "benchmark_results.csv")
    all_metrics = [
        ("PubMedQA", "pubmedqa_local", ["acc,none"]),
        ("MedQA", "medqa_4options", ["acc,none", "acc_norm,none"]),
        ("ARC-Easy", "arc_easy", ["acc,none", "acc_norm,none"]),
        ("PIQA", "piqa", ["acc,none", "acc_norm,none"]),
        ("OpenBookQA", "openbookqa", ["acc,none", "acc_norm,none"]),
    ]
    header = ["Model"]
    for task_label, _, metric_keys in all_metrics:
        for mk in metric_keys:
            col_name = mk.replace(",none", "")
            header.append(f"{task_label} {col_name}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model_name, task_results in results.items():
            row = [model_name]
            for _, task_key, metric_keys in all_metrics:
                for mk in metric_keys:
                    val = ""
                    for task_name, metrics in task_results.items():
                        if task_key in task_name and mk in metrics:
                            v = metrics[mk]
                            if isinstance(v, (int, float)):
                                val = f"{v * 100:.2f}"
                            break
                    row.append(val)
            writer.writerow(row)
    print(f"CSV table saved: {csv_path}")

    plot_radar(labels, model_scores, output_path)


if __name__ == "__main__":
    main()
