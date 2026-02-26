import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # PyTorch built-in data pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_auc_score

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH    = "data/processed/train.xlsx"
TEST_PATH     = "data/processed/test.xlsx"
TARGET_COL    = "NHR_Stress"
DROP_COLS     = ["Participant", "PA_Activity"]

LEARNING_RATE = 0.01
EPOCHS        = 500
L2_LAMBDA     = 0.001
BATCH_SIZE    = 256

# Convert "S"/"NS" labels to +1/-1 — SVM requires numeric labels
def stress_to_pm1(col: pd.Series) -> np.ndarray:
    s = col.astype(str).str.strip().str.upper()
    return np.where(s.isin(["S", "STRESS", "1", "TRUE"]), 1, -1).astype(np.float32)


# Linear SVM: learns a weighted sum of features to classify stress
class LinearSVM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def hinge_loss(scores, y_pm1):
    return torch.mean(torch.clamp(1 - y_pm1 * scores, min=0.0))


def main():

    print("=" * 55)
    print("      STREL STRESS CLASSIFICATION — SVM")
    print("=" * 55)

    train_df = pd.read_excel(TRAIN_PATH)
    test_df  = pd.read_excel(TEST_PATH)

    print(f"\n  Train : {train_df.shape[0]} rows | {train_df['Participant'].nunique()} participants")
    print(f"  Test  : {test_df.shape[0]} rows |  {test_df['Participant'].nunique()} participants")

    train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True)
    test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns], inplace=True)

    y_train = stress_to_pm1(train_df[TARGET_COL])
    y_test  = stress_to_pm1(test_df[TARGET_COL])

    print(f"\n  Target  : {TARGET_COL}")
    print(f"  Train   : {(y_train==1).sum()} Stress  |  {(y_train==-1).sum()} No-Stress")
    print(f"  Test    : {(y_test==1).sum()} Stress  |  {(y_test==-1).sum()} No-Stress")

    X_train_df = train_df.drop(columns=[TARGET_COL])
    X_test_df  = test_df.drop(columns=[TARGET_COL])


# One-hot encode text columns so the model can read them as numbers
    cat_cols = X_train_df.select_dtypes(include=["object", "str"]).columns.tolist()
    for c in cat_cols:
        X_train_df[c] = X_train_df[c].fillna("Unknown").astype(str)
        X_test_df[c]  = X_test_df[c].fillna("Unknown").astype(str)

    X_train_df = pd.get_dummies(X_train_df, columns=cat_cols, drop_first=False)
    X_test_df  = pd.get_dummies(X_test_df,  columns=cat_cols, drop_first=False)
    X_train_df, X_test_df = X_train_df.align(X_test_df, join="left", axis=1, fill_value=0)

    train_medians = X_train_df.median(numeric_only=True)
    X_train_df    = X_train_df.fillna(train_medians)
    X_test_df     = X_test_df.fillna(train_medians)

    X_train = X_train_df.to_numpy(dtype=np.float32)
    X_test  = X_test_df.to_numpy(dtype=np.float32)
    print(f"\n  Features after encoding : {X_train.shape[1]}")


# Normalize so no feature dominates due to scale differences
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Convert to PyTorch tensors — required format for the model
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

    # TensorDataset + DataLoader: handles batching and shuffling automatically
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = LinearSVM(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler: reduces LR by half if loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    print(f"\n{'=' * 55}")
    print(f"  TRAINING")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | L2: {L2_LAMBDA}")
    print(f"  -----------------------------------------------")
    print(f"  {'Epoch':<12} {'Loss':<20} {'Best Loss'}")
    print(f"  -----------------------------------------------")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:        # DataLoader yields batches
            scores = model(X_batch)
            l2_reg = 0.5 * torch.sum(model.linear.weight ** 2)
            loss   = hinge_loss(scores, y_batch) + L2_LAMBDA * l2_reg

            optimizer.zero_grad()   # clear old gradients before each update
            loss.backward()         # backpropagation: compute gradients
            optimizer.step()        # backpropagation: compute gradients

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)                     # adjust LR if loss plateaus

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")  # save best model weights

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:>4}/{EPOCHS}      {avg_loss:<20.4f} {best_loss:.4f}")

    # Load the best saved model before evaluating
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    with torch.no_grad():
        test_scores = model(X_test_t)
        preds_pm1   = torch.sign(test_scores).cpu().numpy().reshape(-1)

    y_test_01 = (y_test   == 1).astype(int)
    preds_01  = (preds_pm1 == 1).astype(int)

    acc = accuracy_score(y_test_01, preds_01)
    cm  = confusion_matrix(y_test_01, preds_01)
    tn, fp, fn, tp = cm.ravel()

    total   = tn + fp + fn + tp
    correct = tn + tp
    wrong   = fp + fn
    specificity = tn / (tn + fp)
    roc_auc     = roc_auc_score(y_test_01, preds_01)

    print(f"\n{'=' * 55}")
    print(f"  RESULTS")
    print(f"{'=' * 55}")
    print(f"  Target Label   : {TARGET_COL}")
    print(f"  Test Accuracy  : {acc * 100:.2f}%  ({correct}/{total} correct, {wrong} wrong)")
    print(f"  Specificity    : {specificity * 100:.2f}%  (how well it identifies No-Stress)")
    print(f"  ROC-AUC        : {roc_auc:.4f}  (1.0 = perfect | 0.5 = random guessing)")

    print(f"""
  CONFUSION MATRIX
  ─────────────────────────────────────────────────
                       Predicted NS   Predicted S
    Actual No Stress       {tn:>5}         {fp:>5}
    Actual Stress          {fn:>5}         {tp:>5}
  ─────────────────────────────────────────────────
    Correct  →  No Stress: {tn}   Stress: {tp}
    Wrong    →  False Alarm: {fp}  |  Missed Stress: {fn}
  ─────────────────────────────────────────────────""")

    print(f"\n  CLASSIFICATION REPORT")
    print(f"  ─────────────────────────────────────────────────")
    print(classification_report(y_test_01, preds_01,
                                 target_names=["No Stress", "Stress"],
                                 digits=4))

    print(f"\n{'=' * 55}\n")


if __name__ == "__main__":
    main()