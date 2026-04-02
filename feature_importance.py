from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


# -------------------------------------------------
# UPDATE THIS PATH TO YOUR ACTUAL EXCEL FILE
# -------------------------------------------------
DATA_PATH = "/Users/lahariappireddy/Downloads/processed.xlsx"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "metrics"

TARGET_COL = "NHR_Stress"
GROUP_COL = "Participant"

DROP_COLS = [
    "PA_Activity",
    "SNS_Stress",
    "NHR_S",
    "NHR_NS",
    "NHR_0_2SD",
    "SNS_S",
    "SNS_NS",
    "SNSindexThreshold",
]

N_SPLITS = 5
TOP_N = 20


def load_data():
    print("Reading file from:", DATA_PATH)

    df = pd.read_excel(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' not found in dataset.")

    df = df.dropna(subset=[TARGET_COL]).copy()

    # Map labels
    label_map = {"NS": 0, "S": 1}
    df[TARGET_COL] = df[TARGET_COL].map(label_map)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    groups = df[GROUP_COL].copy()

    cols_exist = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=[TARGET_COL, GROUP_COL] + cols_exist, errors="ignore").copy()
    y = df[TARGET_COL].copy()

    return X, y, groups


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_cols),
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor, X):
    feature_names = []

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_feature_names)

    feature_names.extend(numeric_cols)
    return feature_names


def run_feature_importance():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_data()

    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Participants: {groups.nunique()}")

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_importance_frames = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\nProcessing Fold {fold}/{N_SPLITS}")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()

        preprocessor = build_preprocessor(X_train)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("rf", rf),
            ]
        )

        model.fit(X_train, y_train)

        fitted_preprocessor = model.named_steps["preprocess"]
        feature_names = get_feature_names(fitted_preprocessor, X_train)

        rf_model = model.named_steps["rf"]
        mdi_importance = rf_model.feature_importances_

        fold_df = pd.DataFrame({
            "feature": feature_names,
            "importance": mdi_importance,
            "fold": fold,
        })

        fold_importance_frames.append(fold_df)

    all_importances = pd.concat(fold_importance_frames, ignore_index=True)

    summary = (
        all_importances
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    csv_path = OUTPUT_DIR / "feature_importance_ranked.csv"
    plot_path = OUTPUT_DIR / "feature_importance_bars.png"

    summary.to_csv(csv_path, index=False)

    print("\nTop 15 Features:")
    print(summary.head(TOP_N).to_string(index=False))

    plot_df = summary.head(TOP_N).iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances (Random Forest, 5-Fold GroupKFold)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved files:")
    print(csv_path)
    print(plot_path)

    return summary


if __name__ == "__main__":
    run_feature_importance()