import os
import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from vae_model import UNetVAE

"""
Оценивает качество генерации спектрограмм в латентном пространстве:
• cosine similarity по MFCC и Spectral Contrast
• без перехода в WAV
"""

load_dotenv()

SPEC_DATA = os.getenv("SPEC_DATA")
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 64))
SEED = int(os.getenv("METRIC_SEED", 42))
PLOT_DIR = os.getenv("EVAL_PLOT_PATH", "artifacts/logs")
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

# Загрузка спектрограмм
real_specs = joblib.load(SPEC_DATA)
print("Реальных спектрограмм:", real_specs.shape[0])

# Загрузка модели
model = UNetVAE(latent_dim=LATENT_DIM, in_ch=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Генерация
with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, LATENT_DIM, device=device)
    specs = model.decode(z).cpu().numpy()  # [B, 1, H, W]
    gen_specs = specs.squeeze(1)

print("Сгенерировано спектрограмм:", gen_specs.shape[0])

SAMPLE_RATE = 22050

def features_mfcc(S):
    S_db = S * 80 - 80
    S_amp = librosa.db_to_amplitude(S_db)
    y = librosa.feature.inverse.mel_to_audio(S_amp, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_iter=16)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    return mfcc.mean(axis=1)

def features_contrast(S):
    S_db = S * 80 - 80
    S_amp = librosa.db_to_amplitude(S_db)
    y = librosa.feature.inverse.mel_to_audio(S_amp, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_iter=16)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    return contrast.mean(axis=1)

# Выборка реальных спектров
idx_real = np.random.choice(real_specs.shape[0], NUM_SAMPLES, replace=False)
real_batch = real_specs[idx_real]

# Извлечение признаков
print("Извлечение MFCC и Contrast...")
mfcc_real = np.array([features_mfcc(s) for s in real_batch])
contrast_real = np.array([features_contrast(s) for s in real_batch])
mfcc_gen = np.array([features_mfcc(s) for s in gen_specs])
contrast_gen = np.array([features_contrast(s) for s in gen_specs])

# Метрики
mfcc_sim = np.mean(np.diag(cosine_similarity(mfcc_real, mfcc_gen)))
contrast_sim = np.mean(np.diag(cosine_similarity(contrast_real, contrast_gen)))

print(f"\n📊 MFCC cosine mean: {mfcc_sim:.4f}")
print(f"📊 Spectral Contrast cosine mean: {contrast_sim:.4f}")

# Визуализация
plt.figure(figsize=(6, 6))
plt.scatter(mfcc_real[:, 0], mfcc_gen[:, 0], alpha=0.6, label="MFCC_0")
plt.scatter(contrast_real[:, 0], contrast_gen[:, 0], alpha=0.6, label="Contrast_0")
plt.xlabel("Реальные признаки")
plt.ylabel("Сгенерированные признаки")
plt.legend()
plt.grid(True)
plt.title("Cosine Similarity в спектральном пространстве")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "latent_similarity_scatter.png"))
plt.close()
print("📈 График сохранён в", os.path.join(PLOT_DIR, "latent_similarity_scatter.png"))
