"""
NHR_Stress Classification Pipeline
Using PyTorch (feature learning) + XGBoost (final prediction)
Target: NHR_Stress (S = Stress, NS = No Stress)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train_df = pd.read_excel("../../data/processed/train.xlsx")
test_df  = pd.read_excel("../../data/processed/test.xlsx")

print(f"Train shape: {train_df.shape}")
print(f"Test  shape: {test_df.shape}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# Columns to drop
DROP_COLS = ["Participant", "PA_Activity", "SNS_Stress"]  # ID + single-value column

# Categorical columns to encode
CAT_COLS = ["Day", "Period", "Profession", "Gender", "Activity4"]

# Target
TARGET = "NHR_Stress"

def preprocess(df, encoders=None, scaler=None, fit=True):
    df = df.copy()
    df.drop(columns=DROP_COLS, inplace=True)

    # Encode target
    y = (df[TARGET] == "S").astype(int).values
    df.drop(columns=[TARGET], inplace=True)

    # Label encode categoricals
    if fit:
        encoders = {}
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col])

    X = df.values.astype(np.float32)

    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, encoders, scaler

X_train, y_train, encoders, scaler = preprocess(train_df, fit=True)
X_test,  y_test,  _,        _      = preprocess(test_df, encoders=encoders, scaler=scaler, fit=False)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
print(f"Feature count: {X_train.shape[1]}")
print(f"Train class balance — S: {y_train.sum()}, NS: {(y_train==0).sum()}")

# ─────────────────────────────────────────────
# 3. PYTORCH FEATURE EXTRACTOR
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training PyTorch Feature Extractor")
print("=" * 60)

INPUT_DIM   = X_train.shape[1]
HIDDEN_DIM  = 128
LATENT_DIM  = 64   # extracted feature size
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, 1)

    def extract(self, x):
        return self.encoder(x)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DataLoaders
X_tr_t = torch.tensor(X_train, dtype=torch.float32)
y_tr_t  = torch.tensor(y_train, dtype=torch.float32)
X_te_t  = torch.tensor(X_test,  dtype=torch.float32)

train_ds     = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model     = FeatureExtractor(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 5 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ── Sanity Check: Evaluate PyTorch alone ──
# If PyTorch performs poorly here, the 64 extracted features
# will be unreliable and XGBoost results should not be trusted.
print("\n── PyTorch Sanity Check ──")
model.eval()
with torch.no_grad():
    logits   = model(X_te_t.to(device)).cpu()
    pt_probs = torch.sigmoid(logits).numpy()
    pt_preds = (pt_probs >= 0.5).astype(int)

pt_acc = accuracy_score(y_test, pt_preds)
pt_auc = roc_auc_score(y_test, pt_probs)
print(f"PyTorch Accuracy : {pt_acc:.4f}")
print(f"PyTorch ROC-AUC  : {pt_auc:.4f}")
print(f"→ PyTorch learned useful patterns — proceeding to feature extraction." if pt_auc > 0.7
      else "→ WARNING: PyTorch AUC is low — extracted features may not be reliable!")

# ─────────────────────────────────────────────
# 4. EXTRACT FEATURES → XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Extracting Features & Training XGBoost")
print("=" * 60)

model.eval()
with torch.no_grad():
    feats_train = model.extract(X_tr_t.to(device)).cpu().numpy()
    feats_test  = model.extract(X_te_t.to(device)).cpu().numpy()

print(f"Original features          : {X_train.shape[1]}")
print(f"PyTorch extracted features : {feats_train.shape[1]}")

# Combine original features + PyTorch extracted features
X_train_combined = np.hstack([X_train, feats_train])
X_test_combined  = np.hstack([X_test,  feats_test])

print(f"Combined features for XGBoost (train): {X_train_combined.shape}")
print(f"Combined features for XGBoost (test):  {X_test_combined.shape}")

# Hyperparameter tuning
param_grid = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth":     [4, 6],
}

xgb_base = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

print("\nRunning GridSearchCV (3-fold)...")
grid_search = GridSearchCV(xgb_base, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
grid_search.fit(X_train_combined, y_train)

print(f"Best XGBoost params : {grid_search.best_params_}")
print(f"Best CV ROC-AUC     : {grid_search.best_score_:.4f}")

xgb_model = grid_search.best_estimator_

# ─────────────────────────────────────────────
# 5. FINAL EVALUATION — XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Final Evaluation — XGBoost")
print("=" * 60)

xgb_probs = xgb_model.predict_proba(X_test_combined)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_probs)

label_names = ["NS (No Stress)", "S (Stress)"]
print(classification_report(y_test, xgb_preds, target_names=label_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, xgb_preds)
print(pd.DataFrame(cm, index=label_names, columns=label_names))

# ─────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Model':<35} {'Accuracy':>10} {'ROC-AUC':>10}")
print("-" * 57)
print(f"{'PyTorch alone (sanity check)':<35} {pt_acc:>10.4f} {pt_auc:>10.4f}")
print(f"{'XGBoost (39 original + 64 PyTorch)':<35} {xgb_acc:>10.4f} {xgb_auc:>10.4f}")
print("=" * 60)
print(f"\nFinal model : XGBoost trained on {X_train_combined.shape[1]} features "
      f"({X_train.shape[1]} original + {feats_train.shape[1]} PyTorch extracted)")