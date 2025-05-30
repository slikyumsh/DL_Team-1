import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchviz import make_dot
from model import ImprovedSpectrogramClassifier


class SpectrogramGroupDataset(Dataset):
    def __init__(self, root_dir, augment=False, noise_std=0.01):
        self.root_dir = root_dir
        self.augment = augment
        self.noise_std = noise_std
        self.group_dirs = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.group_dirs)

    def __getitem__(self, idx):
        group_path = self.group_dirs[idx]

        spectrograms = []
        for i in range(4):
            npy_path = os.path.join(group_path, f"{i}.npy")
            spec = np.load(npy_path)

            if self.augment:
                noise = np.random.normal(0, self.noise_std, spec.shape)
                spec = np.clip(spec + noise, 0.0, 1.0)

            spectrograms.append(spec)

        data_np = np.array(spectrograms)
        data_tensor = torch.from_numpy(data_np).float()

        label_path = os.path.join(group_path, "label.txt")
        with open(label_path, "r") as f:
            label_str = f.read().strip()
        label = 1 if "chainsaw" in label_str.lower() else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor

train_dataset = SpectrogramGroupDataset("dataset/train", augment=True)
val_dataset = SpectrogramGroupDataset("dataset/val", augment=False)

x, y = train_dataset[0]
print(x.shape)
print(y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_labels(dataset_root):
    counts = {"chainsaw": 0, "other": 0}
    for part in ["train"]:
        part_dir = os.path.join(dataset_root, part)
        for group in os.listdir(part_dir):
            group_path = os.path.join(part_dir, group)
            if not os.path.isdir(group_path):
                continue
            label_file = os.path.join(group_path, "label.txt")
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    label = f.read().strip().lower()
                    if "chainsaw" in label:
                        counts["chainsaw"] += 1
                    else:
                        counts["other"] += 1
    return counts

dataset_path = "dataset"
counts = count_labels(dataset_path)
print("Количество групп с лейблом 'chainsaw':", counts["chainsaw"])
print("Количество групп с лейблом 'other':", counts["other"])


pos_weight = torch.tensor([counts["other"] / counts["chainsaw"]], dtype=torch.float32).to(device)

train_dataset = SpectrogramGroupDataset("dataset/train", augment=True)
val_dataset = SpectrogramGroupDataset("dataset/val", augment=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.output_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)

        x = self.net(x)
        x = x.view(B, T, -1)

        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]

        logits = self.output_fc(out).squeeze(1)
        return logits


model = CNNLSTMClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

best_loss = 100.
patience = 3
patience_counter = 0

for epoch in range(300):
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for x, y in train_bar:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        logits = model(x)
        loss = binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_bar.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    with torch.no_grad():
        for x, y in val_bar:
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            loss = binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            val_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)

            val_bar.set_postfix(loss=loss.item())

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    tpr_val = tp / (tp + fn + 1e-6)
    fpr_val = fp / (fp + tn + 1e-6)

    print(f"Epoch {epoch+1}: "
      f"Train loss = {train_loss / len(train_loader):.4f}, "
      f"Val loss = {val_loss / len(val_loader):.4f}, "
      f"F1 Score = {f1:.4f}, "
      f"ROC-AUC = {roc_auc:.4f}, "
      f"FPR = {fpr_val:.4f}, "
      f"TPR = {tpr_val:.4f}")

    if val_loss / len(val_loader) < best_loss:
        best_loss = val_loss / len(val_loader)
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    scheduler.step(val_loss / len(val_loader))
    print("Current LR:", optimizer.param_groups[0]['lr'])

fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_preds)
torch.save(model.state_dict(), "model_weights.pth")

print(f"Final ROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

model.eval()
model = model.to("cpu")
x = torch.randn(1, 4, 128, 128, requires_grad=True)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")