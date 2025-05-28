import os
import joblib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from vae_model import ConvVAE

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
LATENT_DIM = int(os.getenv("LATENT_DIM"))
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
X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 128, 128)
dataset = TensorDataset(X_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()

model = ConvVAE(latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "recon_loss": [],
    "kl_loss": []
}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    beta = min(1.0, epoch / 10)  # KL-аннелинг

    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_l
        total_kl += kl_l

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
            val_loss += loss.item()

    epoch_train_loss = total_loss / len(train_loader)
    epoch_val_loss = val_loss / len(val_loader)
    epoch_recon_loss = total_recon / len(train_loader)
    epoch_kl_loss = total_kl / len(train_loader)
    scheduler.step(epoch_val_loss)

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["recon_loss"].append(epoch_recon_loss)
    history["kl_loss"].append(epoch_kl_loss)

    print(f"Эпоха {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Recon: {epoch_recon_loss:.4f} | KL: {epoch_kl_loss:.4f}")


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
