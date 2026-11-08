# ===- pytorch-lenet-train.py --------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# LeNet traning with PyTorch runtime.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import LeNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150)
args = parser.parse_args()
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

# Initialize model, loss function, and optimizer
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Prepare data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root=str(output_dir / "datasets"), train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train model
epochs = args.epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(
        f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}", flush=True
    )

# Save parameter values independently of Python module paths.
(output_dir / "lenet").mkdir(exist_ok=True)
torch.save(model.state_dict(), output_dir / "lenet/model.pth")
print("Model saved successfully.")
