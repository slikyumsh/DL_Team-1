import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from dotenv import load_dotenv
from vae_model import ConvVAE

load_dotenv()

LATENT_DIM = int(os.getenv("LATENT_DIM"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM"))
INPUT_DIM = int(os.getenv("SPEC_HEIGHT")) * int(os.getenv("SPEC_WIDTH"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES"))
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
OUTPUT_AUDIO = os.getenv("AUDIO_OUTPUT_DIR")
OUTPUT_PLOTS = os.getenv("PLOT_OUTPUT_DIR")

os.makedirs(OUTPUT_AUDIO, exist_ok=True)
os.makedirs(OUTPUT_PLOTS, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство генерации:", device)

model = ConvVAE().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
with torch.no_grad():
    generated = model.decode(z).cpu().numpy()

for i, sample in enumerate(generated):
    spec = sample.reshape((int(os.getenv("SPEC_HEIGHT")), int(os.getenv("SPEC_WIDTH"))))
    S = librosa.db_to_amplitude(spec * 80 - 80)
    y = librosa.istft(S, hop_length=512)

    rms = np.sqrt(np.mean(y ** 2))
    target_rms = 0.1
    if rms > 0:
        y = y * (target_rms / rms)

    audio_path = os.path.join(OUTPUT_AUDIO, f'synthetic_{i}.wav')
    sf.write(audio_path, y, samplerate=22050)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Generated Spectrogram {i}")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOTS, f'spec_{i}.png')
    plt.savefig(plot_path)
    plt.close()

print(f"Сгенерировано {NUM_SAMPLES} звуков и спектрограмм.")