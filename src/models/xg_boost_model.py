"""
NHR_Stress Classification using XGBoost
Trains XGBoost model and evaluates on test data
Target: NHR_Stress (S = Stress, NS = No Stress)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, GroupKFold
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PREPROCESSING FUNCTION
# ─────────────────────────────────────────────
DROP_COLS = [
    "Participant", "PA_Activity", "SNS_Stress",  # ID and alternate labels
    "NHR_S", "NHR_NS", "NHR_0_2SD",              # Derived from NHR_Stress (leakage)
    "SNS_S", "SNS_NS", "SNSindexThreshold"       # Derived from SNS_Stress (leakage)
]
CAT_COLS = ["Day", "Period", "Profession", "Gender", "Activity4"]
TARGET = "NHR_Stress"

def preprocess(df, scaler=None, train_cols=None, fit=True):
    """
    Preprocess data: drop columns, one-hot encode categoricals, scale features
    """
    df = df.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Encode target
    y = (df[TARGET] == "S").astype(int).values
    df.drop(columns=[TARGET], inplace=True)

    # One-hot encode categoricals
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns], drop_first=False)
    
    # Impute missing values
    df = df.fillna(df.median(numeric_only=True))

    if fit:
        train_cols = df.columns.tolist()
    else:
        df = df.reindex(columns=train_cols, fill_value=0)

    X = df.values.astype(np.float32)

    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler, train_cols


def train_and_evaluate_xgboost():
    """
    Train XGBoost model with 5-fold participant-based cross-validation.
    
    Returns:
        dict: Averaged model results including metrics and ROC curve data
    """

    # ─────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    df = pd.read_excel("../../data/processed/processed.xlsx")
    print(f"Dataset shape: {df.shape}, Participants: {df['Participant'].nunique()}")

    # ─────────────────────────────────────────────
    # 2. 5-FOLD CROSS-VALIDATION
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: 5-Fold Cross-Validation (GroupKFold by Participant)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)
    groups = df["Participant"].values

    fold_metrics = []
    fold_details = []
    last_fpr, last_tpr = None, None
    fold_importances = []
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
    }

    # ─────────────────────────────────────────────
    # STEP 1: Find best params via GridSearchCV on full data
    # ─────────────────────────────────────────────
    print("Running GridSearchCV to find best params...")
    X_all, y_all, _, _ = preprocess(df, fit=True)
    xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    grid_search = GridSearchCV(xgb_base, param_grid, cv=gkf.split(df, groups=groups), scoring="roc_auc", n_jobs=-1, verbose=0)
    grid_search.fit(X_all, y_all)
    best_params = grid_search.best_params_
    print(f"Best params: {best_params}")

    # ─────────────────────────────────────────────
    # STEP 2: GroupKFold evaluation with best params
    # ─────────────────────────────────────────────
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        X_train, y_train, scaler, train_cols = preprocess(train_df, fit=True)
        X_test,  y_test,  _,      _          = preprocess(test_df, scaler=scaler, train_cols=train_cols, fit=False)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **best_params)
        model.fit(X_train, y_train)
        fold_importances.append(pd.Series(model.feature_importances_, index=train_cols))

        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        acc       = accuracy_score(y_test, y_preds)
        precision = precision_score(y_test, y_preds, zero_division=0)
        recall    = recall_score(y_test, y_preds, zero_division=0)
        f1        = f1_score(y_test, y_preds, zero_division=0)
        auc       = roc_auc_score(y_test, y_probs)
        fpr, tpr, _ = roc_curve(y_test, y_probs)

        fold_metrics.append((acc, precision, recall, f1, auc))
        last_fpr, last_tpr = fpr, tpr
        tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
        total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp
        fold_details.append({'model': 'XGBoost', 'fold': fold, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': auc})

        print(f"Fold {fold} — Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # ─────────────────────────────────────────────
    # 3. AVERAGE METRICS
    # ─────────────────────────────────────────────
    accs, precs, recs, f1s, aucs = zip(*fold_metrics)
    acc       = np.mean(accs)
    precision = np.mean(precs)
    recall    = np.mean(recs)
    f1        = np.mean(f1s)
    auc       = np.mean(aucs)

    print("\n" + "=" * 60)
    print("MEAN METRICS ACROSS 5 FOLDS")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    return {
        'model_name': 'XGBoost',
        'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1,
        'tn': total_tn, 'fp': total_fp, 'fn': total_fn, 'tp': total_tp,
        'fpr': last_fpr.tolist(), 'tpr': last_tpr.tolist(),
        'fold_details': fold_details,
        'avg_feature_importances': pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False),
    }


if __name__ == "__main__":
    train_and_evaluate_xgboost()
