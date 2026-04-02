"""
Feature Importance Analysis
Extracts and compares feature importances from RF and XGBoost
using the fitted models returned by the model files.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

RESULTS_DIR = Path(__file__).resolve().parent / "../../results/metrics"
TOP_N = 20


def get_importances(result):
    """Pull averaged feature importances directly from model result dict."""
    return result["avg_feature_importances"]


def plot_importances(rf_imp, xgb_imp, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, imp, title, color in zip(
        axes,
        [rf_imp.head(TOP_N), xgb_imp.head(TOP_N)],
        [f"Random Forest — Top {TOP_N} Features", f"XGBoost — Top {TOP_N} Features"],
        ["steelblue", "coral"]
    ):
        imp_sorted = imp.sort_values(ascending=True)
        ax.barh(imp_sorted.index, imp_sorted.values, color=color, alpha=0.85)
        ax.set_xlabel("Feature Importance")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.show()


def plot_combined_importance(rf_imp, xgb_imp, save_path=None):
    """Grouped bar chart overlaying RF and XGBoost importances."""
    top_features = list(dict.fromkeys(
        rf_imp.head(TOP_N).index.tolist() + xgb_imp.head(TOP_N).index.tolist()
    ))

    df_plot = pd.DataFrame({
        "Random Forest": rf_imp.reindex(top_features, fill_value=0),
        "XGBoost":       xgb_imp.reindex(top_features, fill_value=0),
    }).sort_values("Random Forest", ascending=False)

    df_plot.plot(kind="bar", figsize=(16, 7), color=["steelblue", "coral"], alpha=0.85)
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance Comparison — RF vs XGBoost", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Combined plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    from random_forest_model import train_and_evaluate_random_forest
    from xg_boost_model import train_and_evaluate_xgboost

    print("Running Random Forest...")
    rf_result = train_and_evaluate_random_forest()
    rf_imp = get_importances(rf_result)

    print("\nRunning XGBoost...")
    xgb_result = train_and_evaluate_xgboost()
    xgb_imp = get_importances(xgb_result)

    # Save CSVs
    rf_imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        RESULTS_DIR / "rf_feature_importance.csv", index=False)
    xgb_imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        RESULTS_DIR / "xgb_feature_importance.csv", index=False)

    print(f"\nTop {TOP_N} RF features:\n", rf_imp.head(TOP_N).to_string())
    print(f"\nTop {TOP_N} XGBoost features:\n", xgb_imp.head(TOP_N).to_string())

    plot_importances(rf_imp, xgb_imp, save_path=str(RESULTS_DIR / "feature_importance_bars.png"))
    plot_combined_importance(rf_imp, xgb_imp, save_path=str(RESULTS_DIR / "feature_importance_combined.png"))
