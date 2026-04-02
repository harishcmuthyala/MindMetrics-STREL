# Modified from: https://medium.com/@bharataameriya/using-xgboost-in-pytorch-for-enhanced-model-performance-d4c9f9e10225
# Changed: fetch_california_housing → load_breast_cancer (classification)

import torch
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32) 



# Convert dataset to DMatrix for XGBoost
train_dmatrix = xgb.DMatrix(X_train, label=y_train)
test_dmatrix = xgb.DMatrix(X_test, label=y_test)

# Define model parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "eta": 0.1
}

# Train XGBoost model
xgb_model = xgb.train(params, train_dmatrix, num_boost_round=100)



# Make predictions into classes (0 or 1) based on a threshold of 0.5
preds_prob = xgb_model.predict(test_dmatrix)
preds = (preds_prob > 0.5).astype(int)
preds_torch = torch.tensor(preds, dtype=torch.long)


# Evaluation of the model
from sklearn.metrics import accuracy_score  

accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")