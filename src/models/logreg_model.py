from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "../../data/processed/processed.xlsx"

TARGET_COL = "NHR_Stress"
GROUP_COL = "Participant"

DROP_COLS = [
    "Participant", "PA_Activity", "SNS_Stress",  # ID and alternate labels
    "NHR_S", "NHR_NS", "NHR_0_2SD",              # Derived from NHR_Stress (leakage)
    "SNS_S", "SNS_NS", "SNSindexThreshold",      # Derived from SNS_Stress (leakage)
    "HR", "HR_Baseline", "HR_Normalized"         # Correlated to NHR_Stress
]

N_SPLITS = 5
EPOCHS = 50
LR = 0.05


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def load_data(path):
    df = pd.read_excel(path)
    df = df.dropna(subset=[TARGET_COL]).copy()

    groups = df[GROUP_COL].copy()
    df[TARGET_COL] = df[TARGET_COL].map({"NS": 0, "S": 1})
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    y = df[TARGET_COL].copy()
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore").copy()

    return X, y, groups


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def to_numpy(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def train_model(X_train, y_train):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = LogisticRegression(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)

        loss.backward()
        optimizer.step()

        if epoch in [1, 5, 10, 20, 50]:
            print(f"Epoch {epoch:>3} | Loss: {loss.item():.4f}")

    return model


def evaluate_model(model, X_val, y_val):
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probs),
        "confusion_matrix": confusion_matrix(y_val, preds),
        "fpr": roc_curve(y_val, probs)[0],
        "tpr": roc_curve(y_val, probs)[1],
    }

    return metrics


def train_and_evaluate_logistic_regression(selected_features: list = None):
    X, y, groups = load_data(DATA_PATH)

    # Apply selected_features filter if provided (after DROP_COLS)
    if selected_features:
        X = X[[c for c in selected_features if c in X.columns]]

    print("Raw shape:", X.shape)
    print("Target counts:", y.value_counts().to_dict())
    print("Unique participants:", groups.nunique())

    gkf = GroupKFold(n_splits=N_SPLITS)

    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_auc = []
    fold_details = []
    last_fpr, last_tpr = None, None
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"\n========== Fold {fold} ==========")

        X_train_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        preprocessor = build_preprocessor(X_train_raw)

        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        X_train = to_numpy(X_train)
        X_val = to_numpy(X_val)

        print("X_train:", X_train.shape, "X_val:", X_val.shape)

        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)

        print("\n--- Fold Metrics ---")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

        all_accuracy.append(metrics["accuracy"])
        all_precision.append(metrics["precision"])
        all_recall.append(metrics["recall"])
        all_f1.append(metrics["f1"])
        all_auc.append(metrics["roc_auc"])
        last_fpr, last_tpr = metrics["fpr"], metrics["tpr"]
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
        total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp
        fold_details.append({'model': 'Logistic Regression', 'fold': fold, 'accuracy': metrics['accuracy'], 'precision': metrics['precision'], 'recall': metrics['recall'], 'f1_score': metrics['f1'], 'roc_auc': metrics['roc_auc']})

    print("\n==============================")
    print("5-FOLD GROUPKFOLD CV RESULTS")
    print("==============================")
    print(f"Avg Accuracy : {np.mean(all_accuracy):.4f}")
    print(f"Avg Precision: {np.mean(all_precision):.4f}")
    print(f"Avg Recall   : {np.mean(all_recall):.4f}")
    print(f"Avg F1-score : {np.mean(all_f1):.4f}")
    print(f"Avg ROC-AUC  : {np.mean(all_auc):.4f}")

    return {
        "model_name": "Logistic Regression",
        "accuracy": float(np.mean(all_accuracy)), "precision": float(np.mean(all_precision)),
        "recall": float(np.mean(all_recall)), "f1_score": float(np.mean(all_f1)),
        "tn": total_tn, "fp": total_fp, "fn": total_fn, "tp": total_tp,
        "fpr": last_fpr.tolist(), "tpr": last_tpr.tolist(),
        "fold_details": fold_details,
    }


if __name__ == "__main__":
    train_and_evaluate_logistic_regression()