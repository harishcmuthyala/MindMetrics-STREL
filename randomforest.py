"""NHR_Stress Classification using Random Forest
Trains Random Forest model and evaluates on test data
Target: NHR_Stress (S = Stress, NS = No Stress)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
import warnings
warnings.filterwarnings("ignore")


def train_and_evaluate_random_forest():
    """
    Train Random Forest model with preprocessing pipeline and evaluate on test data
    
    Returns:
        dict: Model results including metrics and ROC curve data
    """
    
    # ─────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    train_df = pd.read_excel("/Users/lahariappireddy/Downloads/train.xlsx")
    test_df = pd.read_excel("/Users/lahariappireddy/Downloads/test.xlsx")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # ─────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)
    
    TARGET_COL = "NHR_Stress"
    COLUMNS_TO_DROP = ["Participant", "PA_Activity", "SNS_Stress"]
    LEAKAGE_COLS = ["Stressindex"]
    
    # Drop columns if they exist
    def safe_drop(df, cols):
        cols_exist = [c for c in cols if c in df.columns]
        return df.drop(columns=cols_exist) if cols_exist else df
    
    train_df = safe_drop(train_df, COLUMNS_TO_DROP + LEAKAGE_COLS)
    test_df = safe_drop(test_df, COLUMNS_TO_DROP + LEAKAGE_COLS)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # Convert labels to 0/1
    label_map = {"NS": 0, "S": 1}
    y_train = y_train.map(label_map)
    y_test = y_test.map(label_map)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Train class balance — S: {y_train.sum()}, NS: {(y_train==0).sum()}")
    
    # ─────────────────────────────────────────────
    # 3. TRAINING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training Random Forest")
    print("=" * 60)
    
    # Identify categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Numerical columns: {len(num_cols)}")
    
    # Build preprocessing pipeline
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
    
    model.fit(X_train, y_train)
    print("Model training complete")
    
    # ─────────────────────────────────────────────
    # 4. EVALUATION
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)
    
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, zero_division=0)
    recall = recall_score(y_test, y_preds, zero_division=0)
    f1 = f1_score(y_test, y_preds, zero_division=0)
    auc = roc_auc_score(y_test, y_probs)
    
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    
    label_names = ["NS (No Stress)", "S (Stress)"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_preds, target_names=label_names))
    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_preds), index=label_names, columns=label_names))
    
    # Calculate TPR and FPR for ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    # Return results for comparison
    results = {
        'model_name': 'Random Forest',
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    
    return results


if __name__ == "__main__":
    train_and_evaluate_random_forest()
