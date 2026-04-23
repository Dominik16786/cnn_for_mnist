import yaml
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from train import train, evaluate
from preprocessing import get_dataloaders
from utils import get_device

with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = get_device()

train_loader, test_loader = get_dataloaders(config["batch_size"])

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

if config["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
else:
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

for epoch in range(config["epochs"]):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)

    print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.2f}%")