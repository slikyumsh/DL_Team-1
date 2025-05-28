import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import torch.nn.functional as F
from vae_model import VAE


load_dotenv()

LATENT_DIM = int(os.getenv("LATENT_DIM"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM"))
INPUT_DIM = int(os.getenv("SPEC_HEIGHT")) * int(os.getenv("SPEC_WIDTH"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES"))
MODEL_PATH = os.getenv("VAE_MODEL_PATH")
SPEC_PATH = os.getenv("SPEC_DATA")
PLOT_PATH = os.getenv("EVAL_PLOT_PATH")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство оценки:", device)

real_specs = joblib.load(SPEC_PATH)
real_specs = real_specs[:NUM_SAMPLES]

model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
with torch.no_grad():
    fake_specs = model.decode(z).cpu().numpy()

similarity_matrix = cosine_similarity(real_specs, fake_specs)
mean_similarity = np.mean(np.diag(similarity_matrix))

plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title(f'Cosine Similarity Matrix (Mean Diagonal: {mean_similarity:.4f})')
plt.xlabel('Generated Samples')
plt.ylabel('Real Samples')
plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.savefig(PLOT_PATH)
plt.show()

print(f"Среднее косинусное сходство: {mean_similarity:.4f}")
print(f"Матрица сохранена: {PLOT_PATH}")
