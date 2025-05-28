import os
import librosa
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

"""
Извлекает лог‑Mel‑спектрограммы из аудиофайлов ESC‑50 и сохраняет
их в виде массива 0‒1 (shape: N × 128 × 128) вместе с метками.

1. Mel‑спектрограммы (n_mels = 128) вместо линейного STFT.
2. Статичная шкала амплитуд: диапазон [‑80 дБ, 0 дБ] → нормировка (S+80)/80.
3. Без глобального min‑max: динамика каждого файла сохраняется.
4. Сохранение массива float32, чтобы не терять точность.
"""

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH")
AUDIO_DIR = os.getenv("AUDIO_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
TARGET_SIZE = (int(os.getenv("SPEC_HEIGHT", 128)), int(os.getenv("SPEC_WIDTH", 128)))  # (n_mels, frames)

N_FFT = 1024
HOP_LENGTH = 256
FMIN = 20
FMAX = SAMPLE_RATE // 2
DB_RANGE = 80  # обрезаем до [-80, 0] дБ
print(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

meta = pd.read_csv(CSV_PATH)
classes = sorted(meta["category"].unique())
print(f"Найдено {len(classes)} классов: {classes}")

specs = []
labels = []

for idx, row in meta.iterrows():
    filename = row["filename"]
    label = row["category"]
    filepath = os.path.join(AUDIO_DIR, filename)

    try:
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=TARGET_SIZE[0],
            fmin=FMIN,
            fmax=FMAX,
            power=2.0,
        )

        S_db = librosa.power_to_db(S, ref=1.0)
        S_db = np.clip(S_db, -DB_RANGE, 0)

        S_norm = (S_db + DB_RANGE) / DB_RANGE  # 0 → ‑80 дБ, 1 → 0 дБ

        S_norm = librosa.util.fix_length(S_norm, size=TARGET_SIZE[1], axis=1)

        specs.append(S_norm.astype(np.float32))
        labels.append(label)

    except Exception as e:
        print(f"Ошибка с файлом {filename}: {e}")

specs = np.stack(specs)  # shape: N × 128 × 128
labels = np.array(labels)

print(f"Всего спектрограмм: {specs.shape[0]}")

spec_path = os.path.join(OUTPUT_DIR, "spectrograms.pkl")
label_path = os.path.join(OUTPUT_DIR, "labels.pkl")
joblib.dump(specs, spec_path, compress=3)
joblib.dump(labels, label_path, compress=3)
print(f"Сохранено: {spec_path}, {label_path}")
