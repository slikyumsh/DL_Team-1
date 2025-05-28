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
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))
TARGET_SIZE = (int(os.getenv("SPEC_HEIGHT")), int(os.getenv("SPEC_WIDTH")))

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
categories = df['category'].unique().tolist()
print(f"Обнаружено {len(categories)} классов: {categories}")

spectrograms = []
labels = []

for _, row in df.iterrows():
    fname = row['filename']
    label = row['category']
    path = os.path.join(AUDIO_DIR, fname)

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(S))
        S_db = librosa.util.fix_length(S_db, size=TARGET_SIZE[1], axis=1)
        S_db = S_db[:TARGET_SIZE[0], :]
        spectrograms.append(S_db)
        labels.append(label)
    except Exception as e:
        print(f"Проблема с файлом {fname}: {e}")

spectrograms = np.array(spectrograms)
spectrograms = (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min())
X = spectrograms.reshape((len(spectrograms), -1))
y = np.array(labels)

joblib.dump(X, os.path.join(OUTPUT_DIR, 'spectrograms.pkl'))
joblib.dump(y, os.path.join(OUTPUT_DIR, 'labels.pkl'))
print(f"Сохранено {len(X)} спектрограмм по {len(categories)} классам")
