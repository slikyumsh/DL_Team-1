import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from vae_model import VAE

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
LATENT_DIM = int(os.getenv("LATENT_DIM"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
DATA_PATH = os.getenv("SPEC_DATA")
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
LOG_PATH = os.getenv("VAE_LOG_PATH")
PLOT_PATH = os.getenv("VAE_PLOT_PATH")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)

X = joblib.load(DATA_PATH)
X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

input_dim = X.shape[1]

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

model = VAE(input_dim, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = {"epoch": [], "train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            val_loss += loss.item()

    epoch_train_loss = total_loss / len(train_loader)
    epoch_val_loss = val_loss / len(val_loader)
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)

    print(f"Эпоха {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")

df = pd.DataFrame(history)
df.to_csv(LOG_PATH, index=False)
print(f"Лог обучения сохранён в {LOG_PATH}")

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()
print(f"График потерь сохранён в {PLOT_PATH}")
