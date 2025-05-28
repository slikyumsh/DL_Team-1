import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import librosa

# Пути
REAL_PATH = 'artifacts/spectrograms/spectrograms.pkl'
MODEL_PATH = 'artifacts/models/vae.pth'

LATENT_DIM = 128
INPUT_DIM = 128 * 128
NUM_SAMPLES = 10
SR = 22050

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_decode1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc_decode2 = torch.nn.Linear(hidden_dim, input_dim)

    def decode(self, z):
        h = torch.relu(self.fc_decode1(z))
        return torch.sigmoid(self.fc_decode2(h))

model = VAE(INPUT_DIM, 512, LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
with torch.no_grad():
    generated = model.decode(z).cpu().numpy()

real = joblib.load(REAL_PATH)[:NUM_SAMPLES]

def compute_mfcc(spec):
    S = librosa.db_to_amplitude(spec.reshape(128, 128) * 80 - 80)
    y = librosa.istft(S, hop_length=512)
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13).mean(axis=1)

def compute_contrast(spec):
    S = librosa.db_to_amplitude(spec.reshape(128, 128) * 80 - 80)
    y = librosa.istft(S, hop_length=512)
    return librosa.feature.spectral_contrast(y=y, sr=SR).mean(axis=1)

# MFCC similarity
mfcc_real = np.array([compute_mfcc(s) for s in real])
mfcc_gen = np.array([compute_mfcc(s) for s in generated])
mfcc_sim = cosine_similarity(mfcc_real, mfcc_gen)
print("✅ MFCC cosine mean:", np.mean(np.diag(mfcc_sim)))

# Spectral contrast similarity
sc_real = np.array([compute_contrast(s) for s in real])
sc_gen = np.array([compute_contrast(s) for s in generated])
sc_sim = cosine_similarity(sc_real, sc_gen)
print("✅ Spectral Contrast cosine mean:", np.mean(np.diag(sc_sim)))
