import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# 1) Paths (works from VS Code Run button)
BASE_DIR = Path(__file__).resolve().parent  # points to data/raw
TRAIN_PATH = BASE_DIR / "train.xlsx"
TEST_PATH = BASE_DIR / "test.xlsx"

# 2) Target column and required drop columns
TARGET_COL = "NHR_Stress"  # values: NS / S


DROP_COLS = ["Participant", "PA_Activity", "SNS_Stress"]


def load_and_prepare(path: Path):
    df = pd.read_excel(path)

    # Keep rows where target exists
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Map target labels to 0/1
    df[TARGET_COL] = df[TARGET_COL].map({"NS": 0, "S": 1})
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Drop ONLY specified columns (and then separate X/y)
    df = df.drop(columns=DROP_COLS, errors="ignore")

    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL], errors="ignore").copy()

    return X, y



# 3) Load train/test

X_train_raw, y_train = load_and_prepare(TRAIN_PATH)
X_test_raw, y_test = load_and_prepare(TEST_PATH)

print("Train raw shape:", X_train_raw.shape, "Target counts:", y_train.value_counts().to_dict())
print("Test raw shape :", X_test_raw.shape, "Target counts:", y_test.value_counts().to_dict())



# 4) Preprocessing: impute + encode + scale

cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

X_train = preprocess.fit_transform(X_train_raw)
X_test = preprocess.transform(X_test_raw)

# Convert sparse -> dense if needed
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

print("After preprocessing:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)



# 5) Convert to PyTorch tensors

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)



# 6) Logistic Regression Model (PyTorch)

class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)  # outputs logits

    def forward(self, x):
        return self.linear(x)


model = LogisticRegression(n_features=X_train_t.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)



# 7) Training loop

EPOCHS = 50

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)

    loss.backward()
    optimizer.step()

    if epoch in [1, 5, 10, 20, 50]:
        print(f"Epoch {epoch:>3} | Loss: {loss.item():.4f}")



# 8) Evaluation

model.eval()
with torch.no_grad():
    logits_test = model(X_test_t)
    probs = torch.sigmoid(logits_test).cpu().numpy().ravel()
    preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)
cm = confusion_matrix(y_test, preds)

print("\n--- Test Metrics ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nConfusion Matrix:\n", cm)