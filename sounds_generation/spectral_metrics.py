import os
import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from vae_model import UNetVAE

"""
Оценивает сходство между сгенерированными и реальными Mel‑спектрограммами
(0‒1, 128×128), не переходя в аудио‑домен.
Метрики:
  • cosine по MFCC 
  • cosine по Spectral‑Contrast
Сгенерированные спектры получаем из модели.
"""

load_dotenv()

SPEC_DATA = os.getenv("SPEC_DATA")           
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 64))
SEED = int(os.getenv("METRIC_SEED", 42))

np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

real_specs = joblib.load(SPEC_DATA)      # float32 0‒1
print("Реальных спектрограмм:", real_specs.shape[0])

model = UNetVAE(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, LATENT_DIM, device=device)
    # пустые skip‑тензоры
    s1 = torch.zeros(NUM_SAMPLES, 32, 64, 64, device=device)
    s2 = torch.zeros(NUM_SAMPLES, 64, 32, 32, device=device)
    s3 = torch.zeros(NUM_SAMPLES, 128, 16, 16, device=device)
    gen_specs = model.decode(z, skips=(s1, s2, s3)).cpu().numpy()  # B×1×128×128
    gen_specs = gen_specs.squeeze(1)                              # B×128×128

print("Сгенерировано спектрограмм:", gen_specs.shape[0])

import librosa

SAMPLE_RATE = 22050 


def features_mfcc(S):
    """MFCC из Mel‑спектра (0‒1) — сначала → dB, потом inverse + MFCC."""
    S_db = S * 80 - 80
    S_amp = librosa.db_to_amplitude(S_db)
    y = librosa.feature.inverse.mel_to_audio(
        M=S_amp,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_iter=16,
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    return mfcc.mean(axis=1)


def features_contrast(S):
    S_db = S * 80 - 80
    S_amp = librosa.db_to_amplitude(S_db)
    y = librosa.feature.inverse.mel_to_audio(S_amp, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_iter=16)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    return contrast.mean(axis=1)

idx_real = np.random.choice(real_specs.shape[0], NUM_SAMPLES, replace=False)
real_batch = real_specs[idx_real]

mfcc_real = np.array([features_mfcc(s) for s in real_batch])
contrast_real = np.array([features_contrast(s) for s in real_batch])

mfcc_gen = np.array([features_mfcc(s) for s in gen_specs])
contrast_gen = np.array([features_contrast(s) for s in gen_specs])

mfcc_sim = np.mean(np.diag(cosine_similarity(mfcc_real, mfcc_gen)))
contrast_sim = np.mean(np.diag(cosine_similarity(contrast_real, contrast_gen)))

print(f"MFCC cosine mean: {mfcc_sim:.4f}")
print(f"Spectral Contrast cosine mean: {contrast_sim:.4f}")
