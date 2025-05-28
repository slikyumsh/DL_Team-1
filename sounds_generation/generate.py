import os
import uuid
import torch
import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv
from vae_model import UNetVAE

"""
generate.py — создаёт WAV‑файлы из латентного пространства U‑Net‑VAE.
Дополнение: передаём нулевые skip‑тензоры корректных форм.
"""

load_dotenv()

MODEL_PATH = os.getenv("VAE_MODEL_PATH")
OUT_DIR = os.getenv("GEN_AUDIO_DIR", "generated")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 5))
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
DB_RANGE = 80

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Генерация на устройстве:", device)

model = UNetVAE(latent_dim=LATENT_DIM).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print("Загружена модель:", MODEL_PATH)


def empty_skips(batch: int = 1, device="cpu"):
    s1 = torch.zeros(batch, 32, 64, 64, device=device)
    s2 = torch.zeros(batch, 64, 32, 32, device=device)
    s3 = torch.zeros(batch, 128, 16, 16, device=device)
    return (s1, s2, s3)


with torch.no_grad():
    for _ in range(NUM_SAMPLES):
        z = torch.randn(1, LATENT_DIM, device=device)
        spec_norm = model.decode(z, skips=empty_skips(device=device))  # → 1×1×128×128
        spec_norm = spec_norm[0, 0].cpu().numpy()
        
        S_db = spec_norm * DB_RANGE - DB_RANGE
        S_mel = librosa.db_to_amplitude(S_db)

        y = librosa.feature.inverse.mel_to_audio(
            M=S_mel,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_iter=120,
            fmin=20,
            fmax=SAMPLE_RATE // 2,
            power=2.0,
        )
        y = y / max(1e-5, np.abs(y).max())

        fname = f"sample_{uuid.uuid4().hex[:8]}.wav"
        sf.write(os.path.join(OUT_DIR, fname), y, SAMPLE_RATE)
        print("Сохранён", fname)
