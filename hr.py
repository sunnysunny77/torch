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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SUBSETS = {
    "digits":  {"range": (0, 9),   "num_classes": 10},  
    "upper":   {"range": (10, 35), "num_classes": 26},   
    "lower":   {"range": (36, 61), "num_classes": 26},   
    "all":     {"range": (0, 61),  "num_classes": 62},
}

SUBSET = "upper"
label_min, label_max = SUBSETS[SUBSET]["range"]
NUM_CLASSES = SUBSETS[SUBSET]["num_classes"]

df_train = pd.read_csv("./emnist-byclass-test.csv", header=None)

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

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

model.to(device)

augment = nn.Sequential(
    K.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
    K.RandomContrast(0.05) 
).to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch = augment(X_batch)  

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return acc

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=30)

torch.save(model.state_dict(), "cnn_emnist_upper_weights.pth")

def plot_softmax_predictions(model, loader, device, n=10, num_classes=NUM_CLASSES):
    model.eval()
    plotted = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(1).cpu().numpy()
            probs = probs.cpu().numpy()
            
            for i in range(X_batch.size(0)):
                plt.figure(figsize=(6, 4))
                
                img = X_batch[i].cpu().squeeze()
                plt.subplot(1,2,1)
                plt.imshow(img, cmap='gray')
                plt.title(f"True: {y_batch[i].item()}, Pred: {preds[i]}")
                plt.axis('off')
                
                plt.subplot(1,2,2)
                plt.bar(range(num_classes), probs[i])
                plt.xticks(range(num_classes))
                plt.ylim([0, 1])
                plt.title("Softmax Probabilities")
                
                plt.show()
                
                plotted += 1
                if plotted >= n:
                    return
                    
plot_softmax_predictions(model, val_loader, device, n=10, num_classes=NUM_CLASSES)