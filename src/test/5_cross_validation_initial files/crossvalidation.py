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
    classification_report
)

import warnings
warnings.filterwarnings("ignore")


def random_forest_groupkfold_cv():
    """
    Train and evaluate Random Forest using 5-fold GroupKFold
    based on Participant column.

    Returns:
        dict: Average cross-validation results
    """

    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    df = pd.read_excel("/Users/lahariappireddy/Downloads/train.xlsx")
    print(f"Dataset shape: {df.shape}")

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
        "SNS_S", "SNS_NS", "SNSindexThreshold"
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

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\nFold {fold}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

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

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC   : Not defined")

        fold_results.append({
            "fold": fold,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        })

    # --------------------------------------------------
    # 6. AVERAGE RESULTS
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Average Cross-Validation Results")
    print("=" * 60)

    results_df = pd.DataFrame(fold_results)

    print(results_df)

    print("\nAverage Performance:")
    print(f"Accuracy  : {results_df['accuracy'].mean():.4f}")
    print(f"Precision : {results_df['precision'].mean():.4f}")
    print(f"Recall    : {results_df['recall'].mean():.4f}")
    print(f"F1-Score  : {results_df['f1_score'].mean():.4f}")
    print(f"ROC-AUC   : {results_df['roc_auc'].mean():.4f}")

    return {
        "model_name": "Random Forest with GroupKFold",
        "fold_results": fold_results,
        "average_accuracy": results_df["accuracy"].mean(),
        "average_precision": results_df["precision"].mean(),
        "average_recall": results_df["recall"].mean(),
        "average_f1_score": results_df["f1_score"].mean(),
        "average_roc_auc": results_df["roc_auc"].mean()
    }


if __name__ == "__main__":
    results = random_forest_groupkfold_cv()