from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent / "../../data/processed/processed.xlsx"

DROP_COLS = [
    "PA_Activity", "SNS_Stress",
    "NHR_S", "NHR_NS", "NHR_0_2SD",
    "SNS_S", "SNS_NS", "SNSindexThreshold",
    "HR", "HR_Baseline", "HR_Normalized"   # Correlated to NHR_Stress
]

CAT_COLS = ["Day", "Period", "Profession", "Gender", "Activity4"]
TARGET = "NHR_Stress"
GROUP_COL = "Participant"
BOX = 70


def encode_target(df):
    return (df[TARGET].astype(str).str.strip().str.upper() == "S").astype(int).values


def prepare_dataframe(df):
    df = df.copy()
    drop_cols = [TARGET, GROUP_COL] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)

    return X


def build_preprocessor(X):
    categorical_cols = [c for c in CAT_COLS if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])


def box(title):
    print()
    print(" " + "─" * BOX)
    print(f" │{title:^{BOX-2}}│")
    print(" " + "─" * BOX)


def train_svm(selected_features: list = None):
    print("\nLoading Data\n" + "=" * BOX)
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=[TARGET, GROUP_COL]).copy()

    # Apply selected_features filter if provided (after DROP_COLS)
    if selected_features:
        keep = [GROUP_COL, TARGET] + [f for f in selected_features if f not in [GROUP_COL, TARGET]]
        df = df[[c for c in keep if c in df.columns]]

    y = encode_target(df)
    groups = df[GROUP_COL]

    if groups.nunique() < 5:
        raise ValueError("Need at least 5 participants for GroupKFold")

    X = prepare_dataframe(df)
    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("svm", SVC(class_weight="balanced", probability=True, random_state=42))
    ])

    param_grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10, 100]},
        {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.1, 0.01]}
    ]

    gkf = GroupKFold(n_splits=5)

    print("\nTraining with GroupKFold + GridSearch\n" + "=" * BOX)

    # Step 1: Find best params via GridSearchCV on full data
    grid = GridSearchCV(pipeline, param_grid, cv=gkf.split(X, y, groups=groups), scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    best_params = grid.best_params_
    print(f"\nBest Parameters: {best_params}")

    # Step 2: GroupKFold loop with best params, average metrics
    fold_metrics = []
    fold_details = []
    last_fpr, last_tpr = None, None
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build pipeline with best params
        best_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("svm", SVC(probability=True, random_state=42, class_weight="balanced",
                        kernel=best_params["svm__kernel"], C=best_params["svm__C"],
                        **({} if best_params["svm__kernel"] == "linear" else {"gamma": best_params.get("svm__gamma", "scale")})))
        ])
        best_pipeline.fit(X_train, y_train)

        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        fold_metrics.append((acc, precision, recall, f1, roc_auc))
        last_fpr, last_tpr = fpr, tpr
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp
        fold_details.append({'model': 'SVM', 'fold': fold, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': roc_auc})

        print(f"  Fold {fold} — Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

    accs, precs, recs, f1s, aucs = zip(*fold_metrics)
    acc       = np.mean(accs)
    precision = np.mean(precs)
    recall    = np.mean(recs)
    f1        = np.mean(f1s)
    roc_auc   = np.mean(aucs)

    box("MEAN METRICS ACROSS 5 FOLDS")
    print(f" │ {'Accuracy':<24} {acc:>10.4f} {'':<27}│")
    print(f" │ {'Precision':<24} {precision:>10.4f} {'':<27}│")
    print(f" │ {'Recall (Sensitivity)':<24} {recall:>10.4f} {'':<27}│")
    print(f" │ {'F1-Score':<24} {f1:>10.4f} {'':<27}│")
    print(f" │ {'ROC-AUC':<24} {roc_auc:>10.4f} {'':<27}│")
    print(" " + "─" * BOX)

    return {
        "model_name": "SVM",
        "accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1,
        "tn": total_tn, "fp": total_fp, "fn": total_fn, "tp": total_tp,
        "fpr": last_fpr.tolist(), "tpr": last_tpr.tolist(),
        "fold_details": fold_details,
    }


if __name__ == "__main__":
    train_svm()