"""PyTorch classifier using scikit-learn's built-in Iris data."""
from pathlib import Path
import json, random
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random.seed(42); np.random.seed(42); torch.manual_seed(42)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long); y_test = torch.tensor(y_test, dtype=torch.long)

model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=.01)
for _ in range(200):
    opt.zero_grad(); loss = loss_fn(model(X_train), y_train); loss.backward(); opt.step()
with torch.no_grad():
    accuracy = (model(X_test).argmax(1) == y_test).float().mean().item()
out = Path(__file__).parent / "artifacts"; out.mkdir(exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
(out / "metrics.json").write_text(json.dumps({"test_accuracy": accuracy}, indent=2))
print({"test_accuracy": accuracy})
