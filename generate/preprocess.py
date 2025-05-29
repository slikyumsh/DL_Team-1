import os
import librosa
import numpy as np
import pandas as pd
import joblib
import random
from dotenv import load_dotenv
from tqdm import tqdm

# Загрузка переменных окружения
load_dotenv()

# Пути
CSV_PATH = os.getenv("CSV_PATH")
AUDIO_DIR = os.getenv("AUDIO_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Аудио параметры
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
DURATION_SEC = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION_SEC
SAMPLES_PER_AUDIO = int(os.getenv("SAMPLES_PER_AUDIO", 3))

# Спектрограмма
TARGET_SIZE = (int(os.getenv("SPEC_HEIGHT", 256)), int(os.getenv("SPEC_WIDTH", 256)))
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 20
FMAX = SAMPLE_RATE // 2

# Аугментация
AUGMENT_PROB = 0.3  # вероятность применения аугментации

# Подготовка папки вывода
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загрузка метаданных
meta = pd.read_csv(CSV_PATH)
label_map = {label: idx for idx, label in enumerate(sorted(meta["category"].unique()))}

specs = []
label_ids = []

def augment_audio(y):
    """Простая аугментация аудио"""
    if random.random() < AUGMENT_PROB:
        if random.random() < 0.5:
            y = y + 0.005 * np.random.randn(len(y))  # шум
        if random.random() < 0.5:
            shift = int(0.1 * SAMPLE_RATE)
            y = np.roll(y, shift)
        if random.random() < 0.5:
            steps = random.choice([-2, -1, 1, 2])
            y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)
    return y

def extract_samples(y, sr, num_samples=3, duration_sec=5):
    """Извлечение нескольких сегментов из одного аудиофайла"""
    segment_len = duration_sec * sr
    samples = []

    if len(y) < segment_len:
        y = librosa.util.fix_length(y, segment_len)

    for _ in range(num_samples):
        if len(y) > segment_len:
            start_idx = np.random.randint(0, len(y) - segment_len)
            y_chunk = y[start_idx:start_idx + segment_len]
        else:
            y_chunk = y.copy()

        y_chunk = augment_audio(y_chunk)
        samples.append(y_chunk)

    return samples

for idx, row in tqdm(meta.iterrows(), total=len(meta)):
    filename = row["filename"]
    label = row["category"]
    label_id = label_map[label]
    filepath = os.path.join(AUDIO_DIR, filename)

    try:
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        y_segments = extract_samples(y, SAMPLE_RATE, SAMPLES_PER_AUDIO, DURATION_SEC)

        for y_seg in y_segments:
            S = librosa.feature.melspectrogram(
                y=y_seg,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=TARGET_SIZE[0],
                fmin=FMIN,
                fmax=FMAX,
                power=2.0,
            )

            S_max = np.max(S)
            if S_max > 0:
                S_norm = S / S_max
            else:
                S_norm = np.zeros_like(S)

            S_norm = librosa.util.fix_length(S_norm, size=TARGET_SIZE[1], axis=1)

            specs.append(S_norm.astype(np.float32))
            label_ids.append(label_id)

    except Exception as e:
        print(f"Ошибка с файлом {filename}: {e}")

# Сохраняем
specs = np.stack(specs)
label_ids = np.array(label_ids)

joblib.dump(specs, os.path.join(OUTPUT_DIR, "spectrograms.pkl"), compress=3)
joblib.dump(label_ids, os.path.join(OUTPUT_DIR, "labels.pkl"), compress=3)
joblib.dump(label_map, os.path.join(OUTPUT_DIR, "label_map.pkl"))

print(f"\nСохранено спектрограмм: {specs.shape[0]}, форма одной: {specs.shape[1:]}") 
print(f"Категорий: {len(label_map)} — {list(label_map.keys())}")
