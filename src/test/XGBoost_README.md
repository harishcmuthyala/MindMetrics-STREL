# XGBoost Hybrid Model — PyTorch Feature Engineering + XGBoost

## 🎯 Architecture Philosophy

**Problem:** Traditional ML models are limited to hand-crafted features.  
**Solution:** Use PyTorch as a **feature engineer** to learn non-linear representations, then feed them to XGBoost for final classification.

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

   39 Original Features (preprocessed)
            │
            ├──────────────────────────────┐
            │                              │
            ▼                              │
   ┌────────────────────┐                 │
   │  PyTorch Encoder   │                 │
   │  (Feature Learner) │                 │
   │                    │                 │
   │  39 → 128 → 64     │                 │
   │  • BatchNorm       │                 │
   │  • Dropout         │                 │
   │  • ReLU            │                 │
   └─────────┬──────────┘                 │
             │                            │
             ▼                            │
      64 Learned Features                │
             │                            │
             └──────────┬─────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Concatenate    │
              │  [39 + 64] = 103 │
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  XGBoost Model   │
              │  (Final Predictor)│
              │                  │
              │  • GridSearchCV  │
              │  • 3-fold CV     │
              │  • ROC-AUC opt   │
              └─────────┬────────┘
                        │
                        ▼
                  Predictions
                  (S / NS)
```

---

## 🧠 Why This Architecture?

### PyTorch as Feature Engineer
- **Learns non-linear transformations** that traditional feature engineering can't capture
- **Supervised learning**: Trained on the same classification task, so features are task-relevant
- **Dimensionality expansion**: 39 → 64 features (more representation power)

### XGBoost as Final Classifier
- **Sees everything**: Both original features AND learned representations
- **Tree-based strength**: Captures feature interactions and non-monotonic relationships
- **Robust**: Less prone to overfitting than deep networks on small datasets

### Why NOT Ensemble?
❌ **Averaging PyTorch + XGBoost predictions is redundant** because:
- XGBoost already uses PyTorch's learned features
- This would double-count PyTorch's contribution
- XGBoost has strictly more information than PyTorch alone

---

## 📐 Detailed Architecture

### Stage 1: PyTorch Feature Extractor

```python
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=128, latent_dim=64):
        self.encoder = nn.Sequential(
            nn.Linear(39, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, 1)  # For training only
```

**Training Configuration:**
- **Loss:** BCEWithLogitsLoss (binary cross-entropy)
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** StepLR (decay by 0.5 every 10 epochs)
- **Epochs:** 30
- **Batch Size:** 256

**Sanity Check:**
After training, PyTorch is evaluated standalone. If ROC-AUC < 0.7, a warning is issued because extracted features may be unreliable.

---

### Stage 2: Feature Extraction

```python
# Extract 64-dimensional representations
with torch.no_grad():
    feats_train = model.extract(X_train)  # Shape: (N_train, 64)
    feats_test  = model.extract(X_test)   # Shape: (N_test, 64)

# Combine with original features
X_train_combined = np.hstack([X_train, feats_train])  # (N, 103)
X_test_combined  = np.hstack([X_test,  feats_test])   # (N, 103)
```

---

### Stage 3: XGBoost Training

**Hyperparameter Search (GridSearchCV):**
```python
param_grid = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth":     [4, 6],
}
```

**Cross-Validation:**
- 3-fold CV
- Scoring: ROC-AUC
- Best model selected automatically

---

## 🔄 Data Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ STEP 1: Load Data                                        │
│  • train.xlsx, test.xlsx                                 │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 2: Preprocessing                                    │
│  • Drop: Participant, PA_Activity                        │
│  • Target: NHR_Stress → S=1, NS=0                        │
│  • Encode: 6 categorical features (LabelEncoder)         │
│  • Scale: StandardScaler (fit on train)                  │
│  → Output: 39 features                                   │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 3: Train PyTorch                                    │
│  • 30 epochs, batch=256                                  │
│  • Sanity check: Evaluate standalone performance        │
│  → If AUC < 0.7, warn about feature quality              │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 4: Extract Features                                 │
│  • PyTorch encoder: 39 → 64                              │
│  • Concatenate: [39 original + 64 learned] = 103         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 5: Train XGBoost                                    │
│  • GridSearchCV on 103 features                          │
│  • 3-fold CV, ROC-AUC optimization                       │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 6: Evaluate                                         │
│  • Classification report                                 │
│  • Confusion matrix                                      │
│  • Compare: PyTorch alone vs XGBoost hybrid              │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Breakdown

### Input Features (39 after preprocessing)

| Category          | Features | Examples |
|-------------------|----------|----------|
| **Temporal**      | 2        | Hour, Minute |
| **Demographics**  | 4        | Age, Gender, Profession, BMI |
| **Physiological** | 8        | HR, RR, PNSindex, SNSindex, Stressindex, HR_Baseline, HR_Normalized, SNSindexThreshold |
| **Activity**      | 4        | Cadence, Speed, PA_Percent, PA_Intensity |
| **Sleep**         | 1        | Sleep_Time |
| **Stress Metrics**| 5        | NHR_S, NHR_NS, NHR_0_2SD, SNS_S, SNS_NS |
| **NASA-TLX**      | 6        | Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration |
| **Big Five**      | 5        | Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness |
| **Categorical**   | 6        | Day, Period, Profession, Gender, Activity4, SNS_Stress (all encoded) |

### Learned Features (64)
- Non-linear transformations learned by PyTorch encoder
- Task-specific representations optimized for stress classification

### Combined Features (103)
- 39 original + 64 learned → fed to XGBoost

---

## 🚀 Usage

```bash
cd src/models
python xg_boost_model.py
```

**Prerequisites:**
- `../../data/processed/train.xlsx`
- `../../data/processed/test.xlsx`

**Output:**
```
STEP 1: Loading Data
STEP 2: Preprocessing
STEP 3: Training PyTorch Feature Extractor
  Epoch   5/30 | Loss: 0.6234
  Epoch  10/30 | Loss: 0.5891
  ...
── PyTorch Sanity Check ──
PyTorch Accuracy : 0.7234
PyTorch ROC-AUC  : 0.7891
→ PyTorch learned useful patterns — proceeding to feature extraction.

STEP 4: Extracting Features & Training XGBoost
Best XGBoost params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
Best CV ROC-AUC     : 0.8234

STEP 5: Final Evaluation — XGBoost
              precision    recall  f1-score   support
NS (No Stress)     0.85      0.82      0.83       120
S (Stress)         0.78      0.81      0.79        95

SUMMARY
Model                               Accuracy   ROC-AUC
---------------------------------------------------------
PyTorch alone (sanity check)          0.7234    0.7891
XGBoost (39 original + 64 PyTorch)    0.8156    0.8234
```

---

## 🔍 Key Design Decisions

### 1. Why 64 latent dimensions?
- Balances expressiveness vs overfitting
- ~1.6x expansion from 39 original features
- Empirically tested in prototyping phase

### 2. Why BatchNorm + Dropout?
- **BatchNorm**: Stabilizes training, reduces internal covariate shift
- **Dropout**: Prevents overfitting on small dataset (N=24 participants)

### 3. Why train PyTorch end-to-end?
- Supervised feature learning ensures task relevance
- The classifier head provides training signal to encoder
- Encoder learns discriminative features, not just reconstructions

### 4. Why not use PyTorch predictions directly?
- XGBoost leverages both original + learned features
- Tree-based models excel at feature interactions
- Hybrid approach consistently outperforms either model alone

---

## 📈 Expected Performance

| Metric       | PyTorch Alone | XGBoost Hybrid |
|--------------|---------------|----------------|
| Accuracy     | ~0.70-0.75    | ~0.80-0.85     |
| ROC-AUC      | ~0.75-0.80    | ~0.82-0.88     |
| F1-Score     | ~0.68-0.73    | ~0.78-0.83     |

**Note:** Actual performance varies by fold in person-based CV.

---

## 🛠️ Troubleshooting

### PyTorch AUC < 0.7
**Symptom:** Warning message after Step 3  
**Cause:** PyTorch failed to learn useful patterns  
**Solutions:**
- Increase epochs (30 → 50)
- Adjust learning rate (0.001 → 0.0005)
- Check data preprocessing (scaling, encoding)
- Verify class balance

### XGBoost CV AUC < PyTorch AUC
**Symptom:** XGBoost performs worse than PyTorch alone  
**Cause:** Overfitting or poor hyperparameter choices  
**Solutions:**
- Expand param_grid (more max_depth, n_estimators options)
- Add regularization (min_child_weight, gamma)
- Check for data leakage in CV splits

---

## 📚 References

- **XGBoost Paper**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016
- **Feature Learning**: Bengio et al., "Representation Learning: A Review and New Perspectives", TPAMI 2013
- **STREL Dataset**: IEEE Transactions on Affective Computing, 2025
