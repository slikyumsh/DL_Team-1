import os
import librosa
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv


load_dotenv()

CSV_PATH = os.getenv("CSV_PATH")
AUDIO_DIR = os.getenv("AUDIO_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
TARGET_SIZE = (128, 128)  # (n_mels, frames)

N_FFT = 1024
HOP_LENGTH = 256
FMIN = 20
FMAX = SAMPLE_RATE // 2
DB_RANGE = 80
NOISE_LEVEL = 0.005 

os.makedirs(OUTPUT_DIR, exist_ok=True)

meta = pd.read_csv(CSV_PATH)
classes = sorted(meta["category"].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}

specs = []
labels = []

for idx, row in meta.iterrows():
    filename = row["filename"]
    label_name = row["category"]
    label_idx = class_to_idx[label_name]
    filepath = os.path.join(AUDIO_DIR, filename)

    try:
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

        for augmented in [False, True]:
            y_aug = y.copy()
            if augmented:
                noise = np.random.normal(0, NOISE_LEVEL, size=y.shape)
                y_aug += noise

            S = librosa.feature.melspectrogram(
                y=y_aug,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=TARGET_SIZE[0],
                fmin=FMIN,
                fmax=FMAX,
                power=2.0,
            )

            S_db = librosa.power_to_db(S, ref=np.max)
            S_db = np.clip(S_db, -DB_RANGE, 0)
            S_norm = (S_db + DB_RANGE) / DB_RANGE
            S_norm = librosa.util.fix_length(S_norm, size=TARGET_SIZE[1], axis=1)

            specs.append(S_norm.astype(np.float32))
            labels.append(label_idx)

    except Exception as e:
        print(f"Ошибка с {filename}: {e}")

specs = np.stack(specs)
labels = np.array(labels)

joblib.dump(specs, os.path.join(OUTPUT_DIR, "spectrograms.pkl"), compress=3)
joblib.dump(labels, os.path.join(OUTPUT_DIR, "labels.pkl"), compress=3)
joblib.dump(class_to_idx, os.path.join(OUTPUT_DIR, "class_mapping.pkl"), compress=3)

print(f" Всего спектрограмм (вкл. шум): {len(specs)}")
print("Сохранено в:", OUTPUT_DIR)
