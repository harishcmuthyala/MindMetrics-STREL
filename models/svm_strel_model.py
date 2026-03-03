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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH = "data/processed/train.xlsx"
TEST_PATH  = "data/processed/test.xlsx"
DROP_COLS  = ["Participant", "PA_Activity", "SNS_Stress"]
CAT_COLS   = ["Day", "Period", "Profession", "Gender", "Activity4"]
TARGET     = "NHR_Stress"


# ─────────────────────────────────────────────
# PREPROCESSING FUNCTION
# ─────────────────────────────────────────────
def preprocess(df, scaler=None, fit=True):
    """
    Preprocess data: drop columns, encode categoricals, scale features.

    Args:
        df     : Input dataframe
        scaler : StandardScaler (pass None when fit=True)
        fit    : True for train data, False for test data

    Returns:
        features, labels, scaler
    """
    df = df.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Encode target: S = 1 (Stress), NS = 0 (No Stress)
    y_train = (df[TARGET].astype(str).str.strip().str.upper() == "S").astype(int).values
    df.drop(columns=[TARGET], inplace=True)

    # One-hot encode defined categorical columns
    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

    # Fill any remaining missing values with column median
    df = df.fillna(df.median(numeric_only=True))

    X_train_df = df.values.astype(np.float32)

    # Normalize so no feature dominates due to scale differences
    if fit:
        scaler     = StandardScaler()
        X_train_df = scaler.fit_transform(X_train_df)
    else:
        X_train_df = scaler.transform(X_train_df)

    return X_train_df, y_train, scaler


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_svm():
    """
    Train Linear SVM and evaluate on test data.
    Returns a results dict matching the team pipeline format.
    """

    # ─────────────────────────────────────────────
    # STEP 1: LOAD DATA
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    train_df = pd.read_excel(TRAIN_PATH)
    test_df  = pd.read_excel(TEST_PATH)

    print(f"Train shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")

    # ─────────────────────────────────────────────
    # STEP 2: PREPROCESSING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)

    X_train, y_train, scaler = preprocess(train_df, fit=True)
    X_test,  y_test,  _      = preprocess(test_df, scaler=scaler, fit=False)

    print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}  | y_test  : {y_test.shape}")
    print(f"Train class balance — Stress: {y_train.sum()} | No Stress: {(y_train == 0).sum()}")
    print(f"Test class balance  — Stress: {y_test.sum()}  | No Stress: {(y_test == 0).sum()}")

    # ─────────────────────────────────────────────
    # STEP 3: TRAINING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training SVM")
    print("=" * 60)

    # SVC with probability=True enables predict_proba for ROC-AUC
    # kernel=linear makes it a Linear SVM — fast and interpretable
    # C=1.0 is the standard default regularization value
    model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)
    print(f"Training complete — kernel: linear | C: 1.0")

    # ─────────────────────────────────────────────
    # STEP 4: EVALUATION
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)

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

    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")

    label_names = ["NS (No Stress)", "S (Stress)"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_preds, target_names=label_names))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=label_names, columns=label_names))

    # Return results dict — matches team pipeline format for model comparison
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