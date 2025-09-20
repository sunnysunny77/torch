import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia.augmentation as K
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda")
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50

SUBSETS = {
    "digits":  {"range": (0, 9),   "num_classes": 10},  
    "upper":   {"range": (10, 35), "num_classes": 26},   
    "lower":   {"range": (36, 61), "num_classes": 26},   
    "all":     {"range": (0, 61),  "num_classes": 62},
}

SUBSET = "upper"
label_min, label_max = SUBSETS[SUBSET]["range"]
NUM_CLASSES = SUBSETS[SUBSET]["num_classes"]

df_train = pd.read_csv("/kaggle/input/emnist/emnist-byclass-test.csv", header=None)

X = df_train.drop(columns=[0]).to_numpy()
y = df_train[0].to_numpy()

MASK = (label_min <= y) & (y <= label_max)
X, y = X[MASK], y[MASK] - label_min

X = X.reshape(-1, 1, 28, 28)

X = np.flip(np.rot90(X, k=3, axes=(2, 3)), axis=3)

X = X.astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)
y_train = torch.from_numpy(y_train).long()
y_val = torch.from_numpy(y_val).long()

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),

    nn.Flatten(),
    nn.Linear(128 * 3 * 3, 128),
    nn.ReLU(),
    nn.Linear(128, NUM_CLASSES)  
)

model.to(DEVICE)

augment = nn.Sequential(
    K.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
    K.RandomContrast(0.05) 
).to(DEVICE)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(val_loader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "cnn_emnist_upper_weights.pth")