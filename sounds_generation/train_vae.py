import os
import joblib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from vae_model import UNetVAE  # новая модель

"""
train_vae.py — обучение U‑Net‑VAE на лог‑Mel‑спектрограммах.
• Лосс = 0.5·L1 + 0.5·MSE + β·KL; β разогревается до 0.25 за 30 эпох.
•CosineAnnealingLR.
• Grad‑clip (1.0).
• Сохранение лучшей (по val‑loss) модели.
"""

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 200))
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
DATA_PATH = os.getenv("SPEC_DATA")           
MODEL_PATH = os.getenv("VAE_MODEL_PATH")      
LOG_PATH = os.getenv("VAE_LOG_PATH")          
PLOT_PATH = os.getenv("VAE_PLOT_PATH")         

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)

X = joblib.load(DATA_PATH)  # shape: N × 128 × 128, dtype float32 0‒1
X_tensor = torch.from_numpy(X).unsqueeze(1)  # N × 1 × 128 × 128

train_size = int(0.9 * len(X_tensor))
val_size = len(X_tensor) - train_size
train_ds, val_ds = random_split(TensorDataset(X_tensor), [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

model = UNetVAE(latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def vae_loss(recon_x, x, mu, logvar, beta=0.25):
    l1 = F.l1_loss(recon_x, x, reduction="mean")
    mse = F.mse_loss(recon_x, x, reduction="mean")
    recon_loss = 0.5 * l1 + 0.5 * mse
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

best_val = float("inf")
history = {"epoch": [], "train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    sum_loss = sum_recon = sum_kl = 0.0

    # β‑аннелинг: 0 → 0.25 за 30 эпох
    beta = min(0.25, epoch / 30 * 0.25)

    for (x_batch,) in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        recon, mu, logvar = model(x_batch)
        loss, recon_l, kl_l = vae_loss(recon, x_batch, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sum_loss += loss.item()
        sum_recon += recon_l.item()
        sum_kl += kl_l.item()

    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(device)
            recon, mu, logvar = model(x_batch)
            loss, _, _ = vae_loss(recon, x_batch, mu, logvar, beta)
            val_loss += loss.item()

    train_loss = sum_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    recon_loss = sum_recon / len(train_loader)
    kl_loss = sum_kl / len(train_loader)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["recon_loss"].append(recon_loss)
    history["kl_loss"].append(kl_loss)

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
          f"Recon {recon_loss:.4f} | KL {kl_loss:.4f} | β={beta:.3f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  >> New best model saved to {MODEL_PATH}")

df = pd.DataFrame(history)
df.to_csv(LOG_PATH, index=False)
print("Лог обучения сохранён в", LOG_PATH)

plt.figure(figsize=(9, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train")
plt.plot(df["epoch"], df["val_loss"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("U‑Net‑VAE Training")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()
print("График потерь сохранён в", PLOT_PATH)