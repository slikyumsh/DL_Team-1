import os
import glob
import random
import warnings
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Optional: PESQ
try:
    from pesq import pesq
except ImportError:
    pesq = None

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

GEN_DIR = os.getenv("GEN_AUDIO_DIR", "generated")
REF_AUDIO_DIR = os.getenv("AUDIO_DIR")
CSV_LOG = os.getenv("EVAL_LOG_PATH", "artifacts/logs/eval_metrics.csv")
PLOT_DIR = os.path.dirname(CSV_LOG)
N_REF_SAMPLE = int(os.getenv("N_REF_SAMPLE", 200))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))

os.makedirs(PLOT_DIR, exist_ok=True)

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def mfcc_vector(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfcc.mean(axis=1)

def contrast_vector(y: np.ndarray, sr: int) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    return contrast.mean(axis=1)

def cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - cosine(x, y)

# --- Подготовка референсов ---
ref_files_all = glob.glob(os.path.join(REF_AUDIO_DIR, "*.wav"))
random.shuffle(ref_files_all)
ref_files = ref_files_all[:N_REF_SAMPLE]
print(f"Взято {len(ref_files)} референс‑файлов из {REF_AUDIO_DIR}")

ref_mfcc, ref_contrast = [], []
for p in ref_files:
    y = load_audio(p, SAMPLE_RATE)
    ref_mfcc.append(mfcc_vector(y, SAMPLE_RATE))
    ref_contrast.append(contrast_vector(y, SAMPLE_RATE))
ref_mfcc = np.stack(ref_mfcc)          # K × 20
ref_contrast = np.stack(ref_contrast)  # K × 7

# --- Оценка генераций ---
results: List[Tuple] = []

gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "*.wav")))
assert gen_files, f"Нет WAV‑файлов в {GEN_DIR}"

for g in gen_files:
    y = load_audio(g, SAMPLE_RATE)
    if np.max(np.abs(y)) < 0.01:
        print(f"[!] Пропуск пустого файла: {os.path.basename(g)}")
        continue

    g_mfcc = mfcc_vector(y, SAMPLE_RATE)
    g_con = contrast_vector(y, SAMPLE_RATE)

    # Сходства
    mfcc_sim = float(np.mean([cosine_sim(g_mfcc, r) for r in ref_mfcc]))
    con_sim = float(np.mean([cosine_sim(g_con, r) for r in ref_contrast]))

    # PESQ по ближайшему референсу
    if pesq is not None:
        try:
            dists = [cosine_sim(g_mfcc, r) for r in ref_mfcc]
            best_idx = int(np.argmax(dists))
            ref_y = load_audio(ref_files[best_idx], SAMPLE_RATE)
            y_nb = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=8000)
            ref_nb = librosa.resample(ref_y, orig_sr=SAMPLE_RATE, target_sr=8000)
            pesq_score = pesq(8000, ref_nb, y_nb, "wb")
            ref_match = os.path.basename(ref_files[best_idx])
        except Exception:
            pesq_score = np.nan
            ref_match = "ERROR"
    else:
        pesq_score = np.nan
        ref_match = "N/A"

    results.append((
        os.path.basename(g),
        mfcc_sim,
        con_sim,
        pesq_score,
        ref_match
    ))

    print(f"{g}: MFCC {mfcc_sim:.4f} | Contrast {con_sim:.4f} | PESQ {pesq_score:.3f}")

# --- Сохранение CSV ---
df = pd.DataFrame(results, columns=[
    "file", "mfcc_cosine", "contrast_cosine", "pesq", "matched_reference"
])
df.loc["mean"] = df.mean(numeric_only=True)
print("\nСредние значения:")
print(df.loc["mean"].round(4))

df.to_csv(CSV_LOG, index=False)
print("Лог метрик сохранён:", CSV_LOG)

# --- Визуализация ---
plt.figure(figsize=(12, 5))
plt.hist(df["mfcc_cosine"].dropna(), bins=20, alpha=0.7, label="MFCC Cosine")
plt.hist(df["contrast_cosine"].dropna(), bins=20, alpha=0.7, label="Contrast Cosine")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Распределение cosine similarity")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "hist_cosine_similarity.png"))
plt.close()

# Scatter
plt.figure(figsize=(6, 6))
plt.scatter(df["mfcc_cosine"], df["contrast_cosine"], alpha=0.6)
plt.xlabel("MFCC Cosine")
plt.ylabel("Spectral Contrast Cosine")
plt.grid(True)
plt.title("MFCC vs Contrast Similarity")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "scatter_similarity.png"))
plt.close()

print("Графики сохранены в", PLOT_DIR)
