# 🧠 MindMetrics — ML Stress Prediction on STREL

**Course:** CSCI 6838 – Spring 2026 Capstone  
**Team:** Mind Metrics  

Predicts binary mental-stress states (Stressed / Not-Stressed) for crisis
leaders using the [STREL](https://osf.io/qshv7/) dataset and classical ML
algorithms, with SVM implemented in PyTorch.

---

## 📁 Repo Layout

```
MindMetrics-STREL/
├── config/                  # Central config (paths, seeds, thresholds)
├── data/
│   ├── raw/                 # Original CSV (gitignored — see Setup below)
│   ├── processed/           # Per-fold train/test splits (generated)
│   └── data_dictionary.md   # Full column-by-column documentation
├── notebooks/               # EDA & prototyping notebooks
├── src/
│   ├── data/                # Load → missing → encode → scale
│   ├── validation/          # Person-based CV splits + 6 metrics
│   ├── models/              # LR, RF, XGBoost (sklearn) + SVM (PyTorch)
│   ├── tuning/              # Hyperparameter search
│   ├── evaluation/          # Comparison table, feature importance, plots
│   └── pipeline.py          # End-to-end runner
├── tests/                   # Unit tests (CV leakage, metrics, preprocessing)
├── results/                 # Generated plots, tables, saved models
├── docs/                    # Report, presentation, poster
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone
git clone <repo-url>
cd MindMetrics-STREL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Direct link: https://github.com/UH-ACDC/STREL/raw/main/data/Activity_Stress_data_N24.csv
#    Save it as: data/raw/STREL_raw.csv
```

> **PyTorch install note:** `requirements.txt` installs the CPU build by
> default. If you have a GPU, install PyTorch manually first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🔬 Algorithms

| Model              | Framework      | Owner |
|--------------------|----------------|-------|
| Logistic Regression| scikit-learn   | P1    |
| Random Forest      | scikit-learn   | P2    |
| XGBoost            | xgboost        | P3    |
| SVM                | PyTorch        | P4    |

---

## 📊 Evaluation Metrics

Accuracy · Precision · Recall · F1-Score · ROC-AUC · Specificity  
Validated via **person-based 3-fold CV** (no participant appears in both
train and test).

---

## 📖 Key References

- STREL Paper: *"STREL – Naturalistic Dataset and Methods for Studying
  Mental Stress and Relaxation Patterns in Critical Leading Roles"*,
  IEEE Transactions on Affective Computing, 2025.
- Dataset: https://github.com/UH-ACDC/STREL/blob/main/data/Activity_Stress_data_N24.csv
- Original R code: https://github.com/UH-ACDC/STREL/