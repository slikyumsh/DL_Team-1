import os
import torch
import librosa
import numpy as np
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine

load_dotenv()

MODEL_PATH = os.getenv("VAE_RECON_MODEL_PATH")
DATA_PATH = os.getenv("SPEC_DATA")
LABEL_PATH = os.getenv("LABEL_DATA")
CLASS_MAP_PATH = os.getenv("CLASS_MAPPING_PATH")
OUT_CSV = "artifacts/logs/recon_metrics.csv"

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 50))
DB_RANGE = 80
N_FFT = 1024
HOP_LENGTH = 256

from vae_model import UNetCVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_sim(x, y): return 1 - cosine(x, y)

def mfcc_vec(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfcc.mean(axis=1)

def contrast_vec(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    return contrast.mean(axis=1)

X = joblib.load(DATA_PATH)  # [N, 128, 128]
y = joblib.load(LABEL_PATH)
class_map = {v: k for k, v in joblib.load(CLASS_MAP_PATH).items()}

model = UNetCVAE(latent_dim=LATENT_DIM, in_ch=3, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

results = []

for idx in range(10):
    spec = X[idx]
    label = int(y[idx])
    label_name = class_map[label]

    x_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1x1x128x128
    x_tensor = x_tensor.expand(-1, 3, -1, -1).to(device)
    y_tensor = torch.tensor([label], device=device)

    with torch.no_grad():
        recon, _, _ = model(x_tensor, y_tensor)
        recon_spec = recon[0, 0].cpu().numpy()

    spec_db = spec * DB_RANGE - DB_RANGE
    recon_db = recon_spec * DB_RANGE - DB_RANGE
    S_orig = librosa.db_to_amplitude(spec_db)
    S_recon = librosa.db_to_amplitude(recon_db)

    y_orig = librosa.feature.inverse.mel_to_audio(S_orig, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=N_FFT)
    y_recon = librosa.feature.inverse.mel_to_audio(S_recon, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=N_FFT)

    mse = mean_squared_error(spec_db.flatten(), recon_db.flatten())
    mfcc_sim = cosine_sim(mfcc_vec(y_orig, SAMPLE_RATE), mfcc_vec(y_recon, SAMPLE_RATE))
    con_sim = cosine_sim(contrast_vec(y_orig, SAMPLE_RATE), contrast_vec(y_recon, SAMPLE_RATE))

    results.append((label_name, mse, mfcc_sim, con_sim))


df = pd.DataFrame(results, columns=["class", "mse", "mfcc_cosine", "contrast_cosine"])
df.loc["mean"] = df.mean(numeric_only=True)
df.to_csv(OUT_CSV, index=False)

print("Результаты сохранены в", OUT_CSV)
print(df.loc["mean"].round(4))
