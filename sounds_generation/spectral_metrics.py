import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from dotenv import load_dotenv

load_dotenv()

REAL_PATH = os.getenv("SPEC_DATA")
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
LATENT_DIM = int(os.getenv("LATENT_DIM"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES"))
SPEC_HEIGHT = int(os.getenv("SPEC_HEIGHT"))
SPEC_WIDTH = int(os.getenv("SPEC_WIDTH"))
INPUT_DIM = SPEC_HEIGHT * SPEC_WIDTH
SR = int(os.getenv("SAMPLE_RATE"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decode2 = nn.Linear(hidden_dim, input_dim)

    def decode(self, z):
        h = torch.relu(self.fc_decode1(z))
        return torch.sigmoid(self.fc_decode2(h))

model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
with torch.no_grad():
    generated = model.decode(z).cpu().numpy()

real = joblib.load(REAL_PATH)[:NUM_SAMPLES]

def compute_mfcc(spec):
    S = librosa.db_to_amplitude(spec.reshape(SPEC_HEIGHT, SPEC_WIDTH) * 80 - 80)
    y = librosa.istft(S, hop_length=512)
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13).mean(axis=1)

def compute_contrast(spec):
    S = librosa.db_to_amplitude(spec.reshape(SPEC_HEIGHT, SPEC_WIDTH) * 80 - 80)
    y = librosa.istft(S, hop_length=512)
    return librosa.feature.spectral_contrast(y=y, sr=SR).mean(axis=1)

mfcc_real = np.array([compute_mfcc(s) for s in real])
mfcc_gen = np.array([compute_mfcc(s) for s in generated])
mfcc_sim = cosine_similarity(mfcc_real, mfcc_gen)
print(f"MFCC cosine mean: {np.mean(np.diag(mfcc_sim)):.4f}")

sc_real = np.array([compute_contrast(s) for s in real])
sc_gen = np.array([compute_contrast(s) for s in generated])
sc_sim = cosine_similarity(sc_real, sc_gen)
print(f"Spectral Contrast cosine mean: {np.mean(np.diag(sc_sim)):.4f}")
