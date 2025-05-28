import os
import librosa
import numpy as np
import pandas as pd
import joblib

# Пути
CSV_PATH = '../ESC-50/ESC-50-master/meta/esc50.csv'
AUDIO_DIR = '../ESC-50/ESC-50-master/audio'
OUTPUT_DIR = 'artifacts/spectrograms'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Настройки
TARGET_SIZE = (128, 128)
SELECTED_CATEGORIES = ['rain', 'dog', 'bird']

# Чтение CSV и фильтрация
df = pd.read_csv(CSV_PATH)
df = df[df['category'].isin(SELECTED_CATEGORIES)]

spectrograms = []
for _, row in df.iterrows():
    fname = row['filename']
    path = os.path.join(AUDIO_DIR, fname)
    y, sr = librosa.load(path, sr=22050)
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S))
    S_db = librosa.util.fix_length(S_db, size=TARGET_SIZE[1], axis=1)
    S_db = S_db[:TARGET_SIZE[0], :]
    spectrograms.append(S_db)

spectrograms = np.array(spectrograms)
spectrograms = (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min())
X = spectrograms.reshape((len(spectrograms), -1))

joblib.dump(X, os.path.join(OUTPUT_DIR, 'spectrograms.pkl'))
print(f"Сохранено {len(X)} спектрограмм из классов: {SELECTED_CATEGORIES}")
