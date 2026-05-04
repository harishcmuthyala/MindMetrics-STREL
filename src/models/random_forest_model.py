from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent / "../../data/processed/processed.xlsx"


def train_and_evaluate_random_forest(selected_features: list = None):
    """
    Train and evaluate Random Forest using 5-fold GroupKFold
    based on Participant column.

    Returns:
        dict: Average cross-validation results
    """

    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    df = pd.read_excel(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Apply selected_features filter if provided (after DROP_COLS)
    if selected_features:
        keep = ["Participant", "NHR_Stress"] + [f for f in selected_features if f not in ["Participant", "NHR_Stress"]]
        df = df[[c for c in keep if c in df.columns]]

    # --------------------------------------------------
    # 2. DEFINE TARGET, GROUPS, DROP COLUMNS
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Preparing Data")
    print("=" * 60)

    TARGET_COL = "NHR_Stress"

    COLUMNS_TO_DROP = [
        "PA_Activity", "SNS_Stress",   # alternate labels
        "NHR_S", "NHR_NS", "NHR_0_2SD",
        "SNS_S", "SNS_NS", "SNSindexThreshold",
        "HR", "HR_Baseline", "HR_Normalized"   # Correlated to NHR_Stress
    ]

    # Save Participant separately for grouping
    if "Participant" not in df.columns:
        raise ValueError("Participant column not found in dataset.")

    groups = df["Participant"]

    # Drop only feature-leakage columns, not target yet
    cols_exist = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_exist)

    # Split X and y
    X = df.drop(columns=[TARGET_COL, "Participant"])
    y = df[TARGET_COL]

    # Convert labels to 0/1
    label_map = {"NS": 0, "S": 1}
    y = y.map(label_map)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of unique participants: {groups.nunique()}")

    # --------------------------------------------------
    # 3. IDENTIFY COLUMN TYPES
    # --------------------------------------------------
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Numerical columns: {len(num_cols)}")

    # --------------------------------------------------
    # 4. BUILD PIPELINE
    # --------------------------------------------------
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf)
    ])

    # --------------------------------------------------
    # 5. GROUP K-FOLD CROSS VALIDATION
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: 5-Fold GroupKFold Cross-Validation")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)

    fold_results = []
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\nFold {fold}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        # Accumulate importances across folds
        cat_feature_names = model.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        feature_names = cat_feature_names + num_cols
        fold_importances.append(pd.Series(model.named_steps["rf"].feature_importances_, index=feature_names))

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC needs both classes in the test fold
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC   : Not defined")

        fold_results.append({
            "model": "Random Forest", "fold": fold,
            "accuracy": acc, "precision": precision,
            "recall": recall, "f1_score": f1, "roc_auc": auc,
            "fpr": fpr, "tpr": tpr,
        })

    # --------------------------------------------------
    # 6. AVERAGE RESULTS
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Average Cross-Validation Results")
    print("=" * 60)

    results_df = pd.DataFrame(fold_results)

    print(results_df[['model', 'fold', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])

    print("\nAverage Performance:")
    print(f"Accuracy  : {results_df['accuracy'].mean():.4f}")
    print(f"Precision : {results_df['precision'].mean():.4f}")
    print(f"Recall    : {results_df['recall'].mean():.4f}")
    print(f"F1-Score  : {results_df['f1_score'].mean():.4f}")
    print(f"ROC-AUC   : {results_df['roc_auc'].mean():.4f}")

    return {
        "model_name": "Random Forest",
        "accuracy": results_df["accuracy"].mean(), "precision": results_df["precision"].mean(),
        "recall": results_df["recall"].mean(), "f1_score": results_df["f1_score"].mean(),
        "tn": total_tn, "fp": total_fp, "fn": total_fn, "tp": total_tp,
        "fpr": fold_results[-1]["fpr"].tolist(), "tpr": fold_results[-1]["tpr"].tolist(),
        "fold_details": [{k: v for k, v in f.items() if k not in ("fpr", "tpr")} for f in fold_results],
        "avg_feature_importances": pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False),
    }


if __name__ == "__main__":
    train_and_evaluate_random_forest()