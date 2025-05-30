# --- train_vae.py ---
import os
import joblib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from vae_model import UNetCVAE, Discriminator

load_dotenv()
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 200))
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
DATA_PATH = os.getenv("SPEC_DATA")
LABEL_PATH = os.getenv("LABEL_DATA")
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
LOG_PATH = os.getenv("VAE_LOG_PATH")
PLOT_PATH = os.getenv("VAE_PLOT_PATH")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 50))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

X = joblib.load(DATA_PATH)
y = joblib.load(LABEL_PATH)

X_tensor = torch.from_numpy(X).unsqueeze(1).expand(-1, 3, -1, -1)
y_tensor = torch.from_numpy(y).long()

ds = TensorDataset(X_tensor, y_tensor)
train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

vae = UNetCVAE(latent_dim=LATENT_DIM, in_ch=3, num_classes=NUM_CLASSES).to(device)
disc = Discriminator(in_ch=3).to(device)

opt_vae = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
opt_disc = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE * 0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vae, T_max=EPOCHS)
bce = torch.nn.BCELoss()

# --- Loss ---
def vae_loss(recon_x, x, mu, logvar, beta=0.25):
    recon_loss = F.l1_loss(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

# --- KL контроль ---
beta = 0.01
target_kl = 50.0
kl_tolerance = 0.1

history = {"epoch": [], "train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": [], "gan_loss": [], "beta": []}
best_metrics = {
    "val_loss": float("inf"),
    "recon_loss": float("inf"),
    "kl_loss": float("inf"),
    "gan_loss": float("inf")
}

for epoch in range(1, EPOCHS + 1):
    vae.train()
    disc.train()
    sum_total = sum_recon = sum_kl = sum_gan = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        valid = torch.ones(x_batch.size(0), device=device)
        fake = torch.zeros(x_batch.size(0), device=device)

        recon, mu, logvar = vae(x_batch, y_batch)
        loss_vae, recon_l, kl_l = vae_loss(recon, x_batch, mu, logvar, beta)

        pred_real = disc(x_batch)
        pred_fake = disc(recon.detach())
        loss_disc = bce(pred_real, valid) + bce(pred_fake, fake)
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        pred_fake = disc(recon)
        loss_gan = bce(pred_fake, valid)
        total_loss = loss_vae + 0.01 * loss_gan

        opt_vae.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt_vae.step()

        sum_total += total_loss.item()
        sum_recon += recon_l.item()
        sum_kl += kl_l.item()
        sum_gan += loss_gan.item()

    scheduler.step()

    # --- Валидация ---
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            recon, mu, logvar = vae(x_batch, y_batch)
            loss, _, _ = vae_loss(recon, x_batch, mu, logvar, beta)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    train_loss = sum_total / len(train_loader)
    recon_loss = sum_recon / len(train_loader)
    kl_loss = sum_kl / len(train_loader)
    gan_loss = sum_gan / len(train_loader)

    # --- β update ---
    if kl_loss > target_kl * (1 + kl_tolerance):
        beta *= 1.05
    elif kl_loss < target_kl * (1 - kl_tolerance):
        beta *= 0.95
    beta = float(np.clip(beta, 1e-5, 1.0))

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["recon_loss"].append(recon_loss)
    history["kl_loss"].append(kl_loss)
    history["gan_loss"].append(gan_loss)
    history["beta"].append(beta)

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
          f"Recon {recon_loss:.4f} | KL {kl_loss:.4f} | GAN {gan_loss:.4f} | β={beta:.4f}")

    torch.save(vae.state_dict(), MODEL_PATH.replace(".pth", "_last.pth"))
    metric_values = {
        "val_loss": val_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "gan_loss": gan_loss
    }
    for metric, value in metric_values.items():
        if value < best_metrics[metric]:
            best_metrics[metric] = value
            save_path = MODEL_PATH.replace(".pth", f"_best_{metric}.pth")
            torch.save(vae.state_dict(), save_path)
           
df = pd.DataFrame(history)
df.to_csv(LOG_PATH, index=False)
print("Лог сохранён:", LOG_PATH)

for key in ["train_loss", "val_loss", "recon_loss", "kl_loss", "gan_loss", "beta"]:
    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(key.replace("_", " ").title())
    plt.title(f"{key.replace('_', ' ').title()} per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    out_path = os.path.join(os.path.dirname(PLOT_PATH), f"{key}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"График {key} сохранён:", out_path)
