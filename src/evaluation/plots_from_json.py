"""
Regenerate plots from a saved run JSON file.
Usage:
    python plots_from_json.py <path_to_json>
    python plots_from_json.py ../../results/runs/haris_2026-04-05T12-00-00.json
"""

import matplotlib
matplotlib.use("Agg")

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load_run(json_path: str) -> dict:
    return json.loads(Path(json_path).read_text())


def plot_confusion_matrices(model_metrics: list, save_path: str):
    n = len(model_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, model_metrics):
        cm = np.array([[m.get("TN", 0), m.get("FP", 0)],
                       [m.get("FN", 0), m.get("TP", 0)]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred: No Stress", "Pred: Stress"],
                    yticklabels=["Actual: No Stress", "Actual: Stress"])
        ax.set_title(m["Model"], fontweight="bold")

    plt.suptitle("Confusion Matrices (Summed across 5 folds)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(model_metrics: list, save_path: str):
    df = pd.DataFrame(model_metrics)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["steelblue", "coral", "green", "purple"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        x = range(len(df))
        bars = ax.bar(x, df[metric], color=[colors[i % len(colors)] for i in x], alpha=0.8)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison", fontweight="bold", fontsize=14)
        ax.set_ylim([0, 1.1])
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], rotation=45, ha="right", fontsize=10)
        for bar, v in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curves(model_metrics: list, save_path: str):
    # ROC data isn't in the JSON — skip gracefully
    print(f"Skipped: ROC curves (fpr/tpr not stored in run JSON)")


def plot_fold_accuracy(fold_metrics: list, save_path: str):
    df = pd.DataFrame(fold_metrics)
    colors = ["steelblue", "coral", "green", "purple"]

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(df["model"].unique()):
        model_df = df[df["model"] == model].sort_values("fold")
        plt.plot(model_df["fold"], model_df["accuracy"], marker="o", label=model,
                 color=colors[i % len(colors)], linewidth=2, markersize=7)

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Fold - Model Comparison")
    plt.xticks(range(1, 6))
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single run : python plots_from_json.py <path_to_run_json>")
        print("  All runs   : python plots_from_json.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        runs_dir = Path(__file__).resolve().parent / "../../results/runs"
        jsons = sorted(runs_dir.glob("*.json"))
        if not jsons:
            print("No JSON files found in results/runs/")
            sys.exit(0)
        print(f"Found {len(jsons)} run(s) to process\n")
        for jp in jsons:
            sys.argv = [sys.argv[0], str(jp)]
            # re-run the block below for each
            run = load_run(str(jp))
            stem    = jp.stem
            out_dir = jp.parent / stem
            out_dir.mkdir(exist_ok=True)
            model_metrics = run["output"]["model_metrics"]
            fold_metrics  = run["output"]["fold_metrics"]
            print(f"--- {stem} ---")
            plot_confusion_matrices(model_metrics, str(out_dir / "confusion_matrices.png"))
            plot_model_comparison(model_metrics,   str(out_dir / "model_comparison.png"))
            plot_fold_accuracy(fold_metrics,       str(out_dir / "fold_accuracy.png"))
            plot_roc_curves(model_metrics,         str(out_dir / "roc_curves.png"))
        print("\nAll done.")
    else:
        json_path = sys.argv[1]
        run = load_run(json_path)

        stem    = Path(json_path).stem
        out_dir = Path(json_path).parent / stem
        out_dir.mkdir(exist_ok=True)

        model_metrics = run["output"]["model_metrics"]
        fold_metrics  = run["output"]["fold_metrics"]

        print(f"\nRegenerating plots for: {stem}")
        print(f"User: {run.get('user', '?')} | Features: {len(run.get('features_used', []))}")
        print(f"Notes: {run.get('notes', '')}\n")

        plot_confusion_matrices(model_metrics, str(out_dir / "confusion_matrices.png"))
        plot_model_comparison(model_metrics,   str(out_dir / "model_comparison.png"))
        plot_fold_accuracy(fold_metrics,       str(out_dir / "fold_accuracy.png"))
        plot_roc_curves(model_metrics,         str(out_dir / "roc_curves.png"))

        print("\nDone.")
