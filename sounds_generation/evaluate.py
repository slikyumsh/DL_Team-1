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

# Optional: Frechet Audio Distance
try:
    from pesq import pesq
except ImportError:
    pesq = None  # PESQ not installed

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

"""
Оценивает качество сгенерированных аудио (ген‑дир) относительно
референс‑сетов ESC‑50 по трём группам метрик:
  • MFCC cosine similarity
  • Spectral‑Contrast cosine similarity
  • PESQ 
Результаты сохраняются в CSV и печатаются в консоль.
"""

GEN_DIR = os.getenv("GEN_AUDIO_DIR", "generated")
REF_AUDIO_DIR = os.getenv("AUDIO_DIR")       
CSV_LOG = os.getenv("EVAL_LOG_PATH", "artifacts/logs/eval_metrics.csv")
N_REF_SAMPLE = int(os.getenv("N_REF_SAMPLE", 200))  
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))

os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)

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


ref_files_all = glob.glob(os.path.join(REF_AUDIO_DIR, "*.wav"))
random.shuffle(ref_files_all)
ref_files = ref_files_all[:N_REF_SAMPLE]
print(f"Взято {len(ref_files)} референс‑файлов из {REF_AUDIO_DIR}")

ref_mfcc = []
ref_contrast = []

for p in ref_files:
    y = load_audio(p, SAMPLE_RATE)
    ref_mfcc.append(mfcc_vector(y, SAMPLE_RATE))
    ref_contrast.append(contrast_vector(y, SAMPLE_RATE))

ref_mfcc = np.stack(ref_mfcc)          # K × 20
ref_contrast = np.stack(ref_contrast)  # K × 7

results: List[Tuple[str, float, float, float]] = []

gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "*.wav")))
assert gen_files, f"Нет WAV‑файлов в {GEN_DIR}"

for g in gen_files:
    y = load_audio(g, SAMPLE_RATE)
    g_mfcc = mfcc_vector(y, SAMPLE_RATE)
    g_con = contrast_vector(y, SAMPLE_RATE)


    mfcc_sim = float(np.mean([cosine_sim(g_mfcc, r) for r in ref_mfcc]))
    con_sim = float(np.mean([cosine_sim(g_con, r) for r in ref_contrast]))

    if pesq is not None:
        y_nb = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=8000)
        ref_nb = librosa.resample(ref_files_all[0:1] and load_audio(ref_files_all[0], SAMPLE_RATE), SAMPLE_RATE, 8000)
        try:
            pesq_score = pesq(8000, ref_nb, y_nb, "wb")
        except Exception:
            pesq_score = np.nan
    else:
        pesq_score = np.nan

    results.append((os.path.basename(g), mfcc_sim, con_sim, pesq_score))
    print(f"{g}: MFCC {mfcc_sim:.4f} | Contrast {con_sim:.4f} | PESQ {pesq_score:.3f}")


df = pd.DataFrame(results, columns=["file", "mfcc_cosine", "contrast_cosine", "pesq"])
df.loc["mean"] = df.mean(numeric_only=True)

print("\nСредние значения:")
print(df.loc["mean"].round(4))

df.to_csv(CSV_LOG, index=False)
print("Лог метрик сохранён в", CSV_LOG)