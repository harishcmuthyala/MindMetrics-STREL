# MindMetrics — ML Stress Prediction on STREL

**Course:** CSCI 6838 – Spring 2026 Capstone  
**Team:** Mind Metrics  

Predicts binary mental-stress states (Stressed / Not-Stressed) for crisis
leaders using the [STREL](https://osf.io/qshv7/) dataset. Perform
data preprocessing and implements 4 ML algorithms: Logistic Regression,
Random Forest, XGBoost, and SVM.

---

## 📁 Repo Layout

```
MindMetrics-STREL/
├── data/
│   ├── raw/                 # Original CSV (STREL_raw.csv)
│   └── processed/           # Per-fold train/test splits
├── docs/                    # Feature processing guide
├── notebooks/
│   ├── eda/                 # Exploratory data analysis
│   └── prototyping/         # Model experiments (XGBoost+PyTorch hybrid)
├── src/
│   ├── data/                # Data preprocessing pipeline
│   ├── validation/          # Person-based CV splits
│   ├── models/              # Model implementations
│   └── evaluation/          # Metrics and comparison
├── results/
│   ├── eda/                 # Feature analysis outputs
│   ├── models/              # Saved trained models
│   ├── plots/               # Visualizations
│   └── metrics/             # Performance tables
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

| Model               |
|---------------------|
| Logistic Regression |
| Random Forest       |
| XGBoost             |
| SVM                 |


---

## 📊 Evaluation

**Metrics:** Accuracy · Precision · Recall · F1-Score · ROC-AUC · Specificity

**Validation:** Person-based 3-fold CV using GroupKFold on Participant column
(ensures no participant appears in both train and test sets).

**Features:** 26 total — 5 physiological, 2 motion, 3 NASA-TLX, 4 Big Five
personality, 3 demographic, 3 daily patterns, 4 categorical (encoded), 2
temporal (Hour, Minute).

**Target:** NHR_Stress (binary: S=1, NS=0)

---

## 📖 Key References

- STREL Paper: *"STREL – Naturalistic Dataset and Methods for Studying
  Mental Stress and Relaxation Patterns in Critical Leading Roles"*,
  IEEE Transactions on Affective Computing, 2025.
  https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11185201
- Dataset: https://github.com/UH-ACDC/STREL/blob/main/data/Activity_Stress_data_N24.csv
- Original R code: https://github.com/UH-ACDC/STREL/
