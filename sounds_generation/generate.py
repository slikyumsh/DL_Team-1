import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Параметры
LATENT_DIM = 128
HIDDEN_DIM = 512
INPUT_DIM = 128 * 128
NUM_SAMPLES = 5
MODEL_PATH = 'artifacts/models/vae.pth'
OUTPUT_AUDIO = 'artifacts/audio'
OUTPUT_PLOTS = 'artifacts/plots'
os.makedirs(OUTPUT_AUDIO, exist_ok=True)
os.makedirs(OUTPUT_PLOTS, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decode2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_decode1(z))
        return torch.sigmoid(self.fc_decode2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Загрузка модели
model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Генерация
z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
with torch.no_grad():
    generated = model.decode(z).cpu().numpy()

# Преобразование в спектрограммы и аудио
for i, sample in enumerate(generated):
    spec = sample.reshape((128, 128))
    # Обратное масштабирование к dB (примерно)
    S = librosa.db_to_amplitude(spec * 80 - 80)
    y = librosa.istft(S, hop_length=512)

    # Сохранение аудио
    audio_path = os.path.join(OUTPUT_AUDIO, f'synthetic_{i}.wav')
    sf.write(audio_path, y, samplerate=22050)

    # Сохранение спектрограммы
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Generated Spectrogram {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, f'spec_{i}.png'))
    plt.close()

print(f"Сгенерировано {NUM_SAMPLES} звуков и спектрограмм.")
