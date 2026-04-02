import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/processed.xlsx"

DROP_COLS = [
    "PA_Activity", "SNS_Stress",
    "NHR_S", "NHR_NS", "NHR_0_2SD",
    "SNS_S", "SNS_NS", "SNSindexThreshold"
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


def train_svm_groupkfold():
    print("\nLoading Data\n" + "=" * BOX)
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=[TARGET, GROUP_COL]).copy()

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

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=gkf.split(X, y, groups=groups),
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    print(f"\nBest Parameters: {grid.best_params_}")

    y_pred = cross_val_predict(
        best_model, X, y,
        cv=gkf.split(X, y, groups=groups),
        n_jobs=-1
    )

    y_prob = cross_val_predict(
        best_model, X, y,
        cv=gkf.split(X, y, groups=groups),
        n_jobs=-1,
        method="predict_proba"
    )[:, 1]

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_prob)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    total = tn + fp + fn + tp

    box("MODEL PERFORMANCE")
    print(f" │ {'Accuracy':<24} {acc:>10.4f} {'':<27}│")
    print(f" │ {'Precision':<24} {precision:>10.4f} {'':<27}│")
    print(f" │ {'Recall (Sensitivity)':<24} {recall:>10.4f} {'':<27}│")
    print(f" │ {'Specificity':<24} {specificity:>10.4f} {'':<27}│")
    print(f" │ {'F1-Score':<24} {f1:>10.4f} {'':<27}│")
    print(f" │ {'ROC-AUC':<24} {roc_auc:>10.4f} {'':<27}│")
    print(" " + "─" * BOX)

    box(f"CONFUSION MATRIX (n={total})")
    print(f" │ {'':<20}│{'Pred: No Stress':^21}│{'Pred: Stress':^21}│")
    print(" " + "─" * BOX)

    ns_total = tn + fp
    s_total = fn + tp

    row1 = f"{tn:>5} ({(tn/ns_total*100 if ns_total else 0):5.1f}%)"
    row2 = f"{fp:>5} ({(fp/ns_total*100 if ns_total else 0):5.1f}%)"
    row3 = f"{fn:>5} ({(fn/s_total*100 if s_total else 0):5.1f}%)"
    row4 = f"{tp:>5} ({(tp/s_total*100 if s_total else 0):5.1f}%)"

    print(f" │ {'Actual: No Stress':<20}│{row1:^21}│{row2:^21}│")
    print(f" │ {'Actual: Stress':<20}│{row3:^21}│{row4:^21}│")
    print(" " + "─" * BOX)
    print(f" │ {'Missed Stress:':<16} {fn:<6} {'False Alarms:':<16} {fp:<6} {'Correct:':<10} {tn+tp}/{total:<6}│")
    print(" " + "─" * BOX)

    box("CLASSIFICATION REPORT")
    report = classification_report(
        y, y_pred,
        target_names=["No Stress", "Stress"],
        zero_division=0
    )

    for line in report.split("\n"):
        print(f" │ {line:<{BOX-4}}│")
    print(" " + "─" * BOX)

    print("\nTraining final model on full dataset...\n")
    best_model.fit(X, y)

    return best_model


if __name__ == "__main__":
    model = train_svm_groupkfold()