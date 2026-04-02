"""
NHR_Stress Classification Pipeline
Using Linear SVM (sklearn) for prediction
Target: NHR_Stress (S = Stress, NS = No Stress)
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = "data/processed/train.xlsx"
TEST_PATH  = "data/processed/test.xlsx"
DROP_COLS  = ["Participant", "PA_Activity", "SNS_Stress"]
CAT_COLS   = ["Day", "Period", "Profession", "Gender", "Activity4"]
TARGET     = "NHR_Stress"


def encode_target(df):
    """
    Extract and encode the target column.
    Returns binary array — 1 = Stress, 0 = No Stress
    """
    return (df[TARGET].astype(str).str.strip().str.upper() == "S").astype(int).values


def prepare_features(df, scaler=None, fit=True):
    """
    Build the feature matrix: drop columns, one-hot encode categoricals,
    impute missing values with median, and normalize.

    fit=True  — fits scaler on train data
    fit=False — reuses fitted scaler on test data (prevents leakage)
    """
    df = df.copy()
    df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns], inplace=True)

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns], drop_first=False)

    df = df.fillna(df.median(numeric_only=True))

    feat_names = df.columns.tolist()
    X          = df.values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)
    else:
        X      = scaler.transform(X)

    return X, scaler, feat_names


def train_svm():
    """
    Train Linear SVM and evaluate on test data.
    Returns a results dict matching the team pipeline format.
    """

    print("Loading Data")
    print("=" * 55)
    train_df = pd.read_excel(TRAIN_PATH)
    test_df  = pd.read_excel(TEST_PATH)
    print(f"  Train : {train_df.shape[0]} rows | {train_df.shape[1]} columns")
    print(f"  Test  : {test_df.shape[0]} rows | {test_df.shape[1]} columns")

    print("\nEncoding Labels")
    print("=" * 55)
    y_train = encode_target(train_df)
    y_test  = encode_target(test_df)
    print(f"  Train — Stress: {y_train.sum()} | No Stress: {(y_train == 0).sum()}")
    print(f"  Test  — Stress: {y_test.sum()} | No Stress: {(y_test == 0).sum()}")

    print("\nPreparing Features & Scaling")
    print("=" * 55)
    X_train, scaler, feat_names = prepare_features(train_df, fit=True)
    X_test,  _,      _          = prepare_features(test_df, scaler=scaler, fit=False)
    print(f"  Features after one-hot encoding : {X_train.shape[1]}")
    print(f"  Train matrix : {X_train.shape}")
    print(f"  Test matrix  : {X_test.shape}")

    print("\nTraining SVM")
    print("=" * 55)
    model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)
    print(f"  kernel=linear | C=1.0 | Training complete")

    print("\nEvaluation")
    print("=" * 55)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    acc         = accuracy_score(y_test, y_preds)
    precision   = precision_score(y_test, y_preds)
    recall      = recall_score(y_test, y_preds)
    f1          = f1_score(y_test, y_preds)
    roc_auc     = roc_auc_score(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    cm          = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    total       = tn + fp + fn + tp

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │           MODEL PERFORMANCE              │")
    print("  ├───────────────────────┬─────────────────┤")
    print(f"  │  Accuracy             │     {acc:.4f}      │")
    print(f"  │  Precision            │     {precision:.4f}      │")
    print(f"  │  Recall (Sensitivity) │     {recall:.4f}      │")
    print(f"  │  Specificity          │     {specificity:.4f}      │")
    print(f"  │  F1-Score             │     {f1:.4f}      │")
    print(f"  │  ROC-AUC              │     {roc_auc:.4f}      │")
    print("  └───────────────────────┴─────────────────┘")

    print()
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │             CONFUSION MATRIX  (n={total})              │")
    print(f"  ├──────────────────────┬──────────────┬───────────────┤")
    print(f"  │                      │  Pred: No S  │  Pred: Stress │")
    print(f"  ├──────────────────────┼──────────────┼───────────────┤")
    print(f"  │  Actual: No Stress   │  {tn:>4} ({tn/(tn+fp)*100:4.1f}%) │  {fp:>4} ({fp/(tn+fp)*100:4.1f}%)  │")
    print(f"  │  Actual: Stress      │  {fn:>4} ({fn/(fn+tp)*100:4.1f}%) │  {tp:>4} ({tp/(fn+tp)*100:4.1f}%)  │")
    print(f"  └──────────────────────┴──────────────┴───────────────┘")
    print(f"  ✗ Missed Stress : {fn:<5} ✗ False Alarms : {fp:<5} ✓ Correct : {tn+tp}/{total}")

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │          CLASSIFICATION REPORT           │")
    print("  └─────────────────────────────────────────┘")
    label_names = ["NS (No Stress)", "S (Stress)"]
    print(classification_report(y_test, y_preds, target_names=label_names))

    return {
        "model_name" : "SVM",
        "accuracy"   : acc,
        "precision"  : precision,
        "recall"     : recall,
        "f1_score"   : f1,
        "specificity": specificity,
        "roc_auc"    : roc_auc,
        "fpr"        : fpr.tolist(),
        "tpr"        : tpr.tolist(),
    }


if __name__ == "__main__":
    train_svm()