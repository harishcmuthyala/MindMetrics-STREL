import numpy as np
import xgboost as xgb
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the dataset
data = load_wine()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train an XGBoost model on the dataset
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Generate predictions (or raw scores) from XGBoost
train_xgb_preds = xgb_model.predict_proba(X_train)  # Probabilities for each class
test_xgb_preds = xgb_model.predict_proba(X_test)

# Step 3: Combine XGBoost predictions (probabilities) with original features
X_train_combined = np.hstack((X_train, train_xgb_preds))
X_test_combined = np.hstack((X_test, test_xgb_preds))

# Step 4: Convert combined data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 5: Define the PyTorch neural network for classification
class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(np.unique(y)))  # Output layer size is number of classes
        )

    def forward(self, x):
        return self.fc(x)

# Step 6: Initialize the model, loss function, and optimizer
input_dim = X_train_combined.shape[1]  # Combined features (original + XGBoost predictions)
model = HybridModel(input_dim)

criterion = nn.CrossEntropyLoss()  # For classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 7: Train the PyTorch model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Step 8: Evaluate the model
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor)
    test_predictions = model(X_test_tensor)

    train_pred_labels = torch.argmax(train_predictions, dim=1)
    test_pred_labels = torch.argmax(test_predictions, dim=1)

    train_accuracy = accuracy_score(y_train, train_pred_labels)
    test_accuracy = accuracy_score(y_test, test_pred_labels)

print(f"Final Training Accuracy: {train_accuracy:.4f}")
print(f"Final Testing Accuracy: {test_accuracy:.4f}")