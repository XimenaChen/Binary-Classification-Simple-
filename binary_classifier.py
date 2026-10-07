import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


# =========================
# 1. Prepare Data
# =========================

torch.manual_seed(42)

X = torch.randn(1000, 2)

y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

# Split data
train_X = X[:800]
train_y = y[:800]

test_X = X[800:]
test_y = y[800:]

#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

train_dataset = TensorDataset(train_X, train_y)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# =========================
# 2. Define Model
# =========================

class BinaryClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(2,1)

    def forward(self, x):

        return self.linear(x)


# =========================
# 3. Model / Loss / Optimizer
# =========================

model = BinaryClassifier()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam( #二分法用这个
    model.parameters(),
    lr=0.01
)

# =========================
# 4. Training
# =========================

for epoch in range(100):

    for X_batch, y_batch in train_loader:

        # Forward
        prediction = model(X_batch)

        # Loss
        loss = criterion(prediction, y_batch)

        # Clear gradients
        optimizer.zero_grad()

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()

# =========================
# 5. Evaluation
# =========================

model.eval()

with torch.no_grad():

    logits = model(test_X)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > 0.5).float()

    accuracy = (predictions == test_y).float().mean()

print("Accuracy:", accuracy.item())

# =========================
# 6. Save Model
# =========================
'''
torch.save(
    model.state_dict(),
    "binary_classifier.pth"
)'''