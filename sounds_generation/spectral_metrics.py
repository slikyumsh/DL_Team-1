import os
import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from dotenv import load_dotenv
from vae_model import ConvVAE

load_dotenv()

REAL_PATH = os.getenv("SPEC_DATA")
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
LATENT_DIM = int(os.getenv("LATENT_DIM"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM"))  # необязательно, но на всякий случай
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES"))
SPEC_HEIGHT = int(os.getenv("SPEC_HEIGHT"))
SPEC_WIDTH = int(os.getenv("SPEC_WIDTH"))
SR = int(os.getenv("SAMPLE_RATE"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

# Загрузка модели
model = ConvVAE(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Генерация данных
z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
with torch.no_grad():
    generated = model.decode(z).cpu().numpy()

# Загрузка реальных спектрограмм
real = joblib.load(REAL_PATH)[:NUM_SAMPLES]

# Метрики
def compute_mfcc(spec):
    S = librosa.db_to_amplitude(spec.reshape(SPEC_HEIGHT, SPEC_WIDTH) * 80 - 80)
    y = librosa.griffinlim(S, hop_length=512)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y *= 0.1 / rms
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13).mean(axis=1)

def compute_contrast(spec):
    S = librosa.db_to_amplitude(spec.reshape(SPEC_HEIGHT, SPEC_WIDTH) * 80 - 80)
    y = librosa.griffinlim(S, hop_length=512)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y *= 0.1 / rms
    return librosa.feature.spectral_contrast(y=y, sr=SR).mean(axis=1)

# Вычисление сходства
mfcc_real = np.array([compute_mfcc(s) for s in real])
mfcc_gen = np.array([compute_mfcc(s) for s in generated])
mfcc_sim = cosine_similarity(mfcc_real, mfcc_gen)
print(f"MFCC cosine mean: {np.mean(np.diag(mfcc_sim)):.4f}")

sc_real = np.array([compute_contrast(s) for s in real])
sc_gen = np.array([compute_contrast(s) for s in generated])
sc_sim = cosine_similarity(sc_real, sc_gen)
print(f"Spectral Contrast cosine mean: {np.mean(np.diag(sc_sim)):.4f}")
