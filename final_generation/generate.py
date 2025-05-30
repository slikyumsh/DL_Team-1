import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
import soundfile as sf
from dotenv import load_dotenv
from vae_model import UNetCVAE

# === Константы ===
load_dotenv()
MODEL_PATH      = os.getenv("VAE_MODEL_PATH")
CLASS_MAP_PATH  = os.getenv("CLASS_MAPPING_PATH")
SPEC_DATA_PATH  = os.getenv("SPEC_DATA")
LABEL_DATA_PATH = os.getenv("LABEL_DATA")
OUT_DIR         = os.getenv("GEN_AUDIO_DIR", "generated")
LATENT_DIM      = int(os.getenv("LATENT_DIM", 128))
SAMPLE_RATE     = int(os.getenv("SAMPLE_RATE", 22050))
NUM_CLASSES     = int(os.getenv("NUM_CLASSES", 50))

DB_RANGE   = 80
N_FFT      = 1024
HOP_LENGTH = 256
NUM_SAMPLES = 5 

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

model = UNetCVAE(latent_dim=LATENT_DIM, in_ch=3, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

idx_to_class = {v: k for k, v in joblib.load(CLASS_MAP_PATH).items()}
X = joblib.load(SPEC_DATA_PATH)
Y = joblib.load(LABEL_DATA_PATH)
X = torch.from_numpy(X).unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, 128, 128)
Y = torch.from_numpy(Y).long()

with torch.no_grad():
    indices = torch.randperm(X.shape[0])[:NUM_SAMPLES]
    x_samples = X[indices].to(device)
    y_samples = Y[indices].to(device)

    mu, logvar, skips, y_onehot = model.encode(x_samples, y_samples)
    z = mu + 0.1 * torch.randn_like(mu)
    recon = model.decode(z, skips, y_onehot)

    for i in range(NUM_SAMPLES):
        class_name = idx_to_class[int(y_samples[i].item())]

        orig_mel = x_samples[i, 0].cpu().numpy() * DB_RANGE - DB_RANGE
        recon_mel = recon[i, 0].cpu().numpy() * DB_RANGE - DB_RANGE

        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        librosa.display.specshow(orig_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 y_axis="mel", x_axis="time", cmap="magma", ax=axs[0])
        axs[0].set_title(f"Original [{class_name}]")

        librosa.display.specshow(recon_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 y_axis="mel", x_axis="time", cmap="magma", ax=axs[1])
        axs[1].set_title(f"Generated")

        for ax in axs:
            ax.label_outer()
        plt.tight_layout()
        out_img = os.path.join(OUT_DIR, f"compare_{class_name}_{i}.png")
        plt.savefig(out_img)
        plt.close()
        print("Сохранена спектрограмма:", out_img)

        S_recon = librosa.db_to_amplitude(recon_mel)
        y_audio = librosa.feature.inverse.mel_to_audio(
            M=S_recon, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_iter=1000,
            fmin=20, fmax=SAMPLE_RATE // 2, power=2
        )
        sf.write(os.path.join(OUT_DIR, f"recon_{class_name}_{i}.wav"), y_audio, SAMPLE_RATE)
