import os
import uuid
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import joblib
from dotenv import load_dotenv
from vae_model import UNetVAE
from skimage.metrics import structural_similarity as ssim

# Загрузка переменных среды
load_dotenv()

# Конфигурация
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
DATA_PATH = os.getenv("SPEC_DATA")
OUT_DIR = os.getenv("GEN_AUDIO_DIR", "generated")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 5))
LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
TARGET_CLASS = os.getenv("TARGET_CLASS", None)

# Спектр параметры
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 256
ENERGY_THRESHOLD = 0.05  # фильтрация «пустых» спектров

# Подготовка
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Генерация на устройстве:", device)

# Модель
model = UNetVAE(latent_dim=LATENT_DIM, in_ch=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Загружена модель:", MODEL_PATH)

# Загрузка данных
data = joblib.load(DATA_PATH)
if isinstance(data, tuple) and len(data) == 2:
    X, y = data
    if TARGET_CLASS is not None:
        mask = (y == int(TARGET_CLASS))
        X = X[mask]
        y = y[mask]
        print(f"Отфильтровано по классу {TARGET_CLASS}: {len(X)} примеров")
else:
    X = data

# Фильтрация по амплитуде
valid_idx = [i for i in range(X.shape[0]) if X[i].max() >= ENERGY_THRESHOLD]
X = X[valid_idx]
print(f"Пропущено {len(data[0]) - len(X)} из-за низкой энергии")

# Подготовка батча
X_tensor = torch.from_numpy(X).unsqueeze(1).to(device)

# Генерация
with torch.no_grad():
    idx = torch.randint(0, X_tensor.size(0), (NUM_SAMPLES,))
    selected = X_tensor[idx]
    mu, logvar = model.encode(selected)
    z = model.reparameterize(mu, logvar)
    generated = model.decode(z)

    for i in range(NUM_SAMPLES):
        real_spec = selected[i][0].cpu().numpy()
        gen_spec = generated[i][0].cpu().numpy()

        # Визуализация: сравнение
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(real_spec, aspect="auto", origin="lower")
        axs[0].set_title(f"Real Spectrogram {i+1}")
        axs[1].imshow(gen_spec, aspect="auto", origin="lower")
        axs[1].set_title(f"Generated Spectrogram {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"compare_{i+1}.png"))
        plt.close()

        # По отдельности
        plt.imshow(real_spec, aspect="auto", origin="lower")
        plt.title(f"Real Spectrogram {i+1}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"real_{i+1}.png"))
        plt.close()

        plt.imshow(gen_spec, aspect="auto", origin="lower")
        plt.title(f"Generated Spectrogram {i+1}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"gen_{i+1}.png"))
        plt.close()

        # SSIM
        score = ssim(real_spec, gen_spec, data_range=1.0)
        print(f"SSIM {i+1}: {score:.4f}")

        # Инверсия в аудио
        S_amp = librosa.db_to_amplitude(gen_spec * 80 - 80)
        y = librosa.feature.inverse.mel_to_audio(
            M=S_amp,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            fmin=20,
            fmax=SAMPLE_RATE // 2,
            power=2.0,
            n_iter=1000,
        )
        y = y / max(1e-5, np.abs(y).max())

        fname = f"sample_{uuid.uuid4().hex[:8]}.wav"
        sf.write(os.path.join(OUT_DIR, fname), y, SAMPLE_RATE)
        print("Сохранён:", fname)
