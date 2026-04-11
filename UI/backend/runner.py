import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/evaluation'))

from model_comparison import run_all

METRICS_DIR = os.path.join(os.path.dirname(__file__), '../../results/metrics')


def run_model(selected_features: list[str]) -> dict:
    run_all(selected_features=selected_features)

    model_metrics = pd.read_csv(f"{METRICS_DIR}/model_metrics.csv").to_dict(orient='records')
    fold_metrics  = pd.read_csv(f"{METRICS_DIR}/fold_metrics.csv").to_dict(orient='records')

    return {
        "selected_features": selected_features,
        "model_metrics": model_metrics,
        "fold_metrics": fold_metrics,
    }
