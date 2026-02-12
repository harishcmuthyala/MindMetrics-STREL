import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Load dataset

dataset = load_breast_cancer()
features = dataset.data
labels_binary = dataset.target   # 0 or 1

# Convert labels to -1 and +1 (required for SVM hinge loss)
labels = torch.where(torch.tensor(labels_binary) == 0, -1, 1).numpy()


# Separate it as train & test

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)


# Normalize features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Linear SVM model

class SimpleSVM(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.line = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.line(x)

model = SimpleSVM(X_train.shape[1])


# Hinge Loss

def hinge_loss(predictions, true_labels):
    return torch.mean(torch.clamp(1 - true_labels * predictions, min=0))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training loop

epochs = 300

for i in range(epochs):
    outputs = model(X_train_tensor)

    loss = hinge_loss(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 50 epochs so you know it's running
    if (i + 1) % 50 == 0:
        print(f"Epoch {i+1}/{epochs} | Loss: {loss.item():.4f}")


# Testing

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted_labels = torch.sign(test_outputs)  # -1 or +1

# Convert back to 0/1 for accuracy
predicted_01 = (predicted_labels.squeeze().numpy() == 1).astype(int)
true_01 = (y_test_tensor.squeeze().numpy() == 1).astype(int)

accuracy = accuracy_score(true_01, predicted_01)

print(f"Accuracy: {accuracy*100:.2f}%")
print("First 10 predictions (+1/-1):", predicted_labels[:10].view(-1).int().tolist())
print("First 10 true labels (+1/-1):", y_test_tensor[:10].view(-1).int().tolist())