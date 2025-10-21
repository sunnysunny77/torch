import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia.augmentation as K
import torch.optim as optim  
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

lr = 1e-3
batch_size = 64
num_epochs = 50

subsets = {
    "digits": {
        "range": (0, 9),
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)]
    },
    "upper": {
        "range": (10, 35),
        "num_classes": 26,
        "class_names": [chr(i) for i in range(ord('A'), ord('Z')+1)]
    },
    "lower": {
        "range": (36, 61),
        "num_classes": 26,
        "class_names": [chr(i) for i in range(ord('a'), ord('z')+1)]
    },
    "all": {
        "range": (0, 61),
        "num_classes": 62,
        "class_names": [str(i) for i in range(10)] +
                       [chr(i) for i in range(ord('A'), ord('Z')+1)] +
                       [chr(i) for i in range(ord('a'), ord('z')+1)]
    }
}

subset = "upper"
label_min, label_max = subsets[subset]["range"]
num_classes = subsets[subset]["num_classes"]
class_names = subsets[subset]["class_names"]

df_train = pd.read_csv("./emnist-byclass-train.csv", header=None)
df_test = pd.read_csv("./emnist-byclass-test.csv", header=None)

X_train = df_train.drop(columns=[0]).to_numpy()
y_train = df_train[0].to_numpy()

X_test = df_test.drop(columns=[0]).to_numpy()
y_test = df_test[0].to_numpy()

mask_train = (label_min <= y_train) & (y_train <= label_max)
X_train, y_train = X_train[mask_train], y_train[mask_train] - label_min
mask_test = (label_min <= y_test) & (y_test <= label_max)
X_test, y_test = X_test[mask_test], y_test[mask_test] - label_min

def prep(X):
    X = X.reshape(-1, 1, 28, 28)
    X = np.flip(np.rot90(X, k=3, axes=(2, 3)), axis=3)
    X = X.astype(np.float32) / 255.0
    return X
    
X_train = prep(X_train)
X_test = prep(X_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).long()
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val).long()
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).long()

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4(x))
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Augment(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = K.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05))
        self.contrast = K.RandomContrast(0.05)
        
    def forward(self, x):
        if self.training:
            x = self.affine(x)
            x = self.contrast(x)
        return x

model = CNN(num_classes).to(device)
augment = Augment().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = accuracy_score(all_labels, all_preds)

    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Epoch {epoch+1}/{num_epochs} | "f"Train Loss: {train_loss:.6f}, Acc: {train_acc*100:.2f}% | "f"Val Loss: {val_loss:.6f}, Acc: {val_acc*100:.2f}%")

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(y.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=class_names)
acc = accuracy_score(y_true, y_pred)
mcm_test = multilabel_confusion_matrix(y_true, y_pred)

print(report)
print(f"Overall Accuracy: {acc*100:.2f}%")
for i, cm in enumerate(mcm_test):
    tn, fp, fn, tp = cm.ravel()
    print(f"Class {i} ({class_names[i]}): TP={tp}, FP={fp}, TN={tn}, FN={fn}")