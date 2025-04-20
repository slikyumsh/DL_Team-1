import torch
from torch.utils.data import Dataset
import numpy as np
import os
from dotenv import load_dotenv

model_config = "./model.env"
load_dotenv(model_config)
load_dotenv(os.getenv("PREPROCESS_PATH"))

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.preproc_dir = os.path.join(os.getenv("DATA_ROOT"), "processed")
        self.data_dir = os.path.join(self.preproc_dir, root_dir)
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for label in self.classes:
            class_dir = os.path.join(self.data_dir, label)
            for fname in os.listdir(class_dir):
                self.samples.append(os.path.join(class_dir, fname))
                self.labels.append(self.class_to_idx[label])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        spectrogram = np.load(self.samples[idx])
        if spectrogram.ndim == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Добавить канал: (1, H, W)
        label = self.labels[idx]
        label = self.labels[idx]
        return torch.tensor(spectrogram, dtype=torch.float16), label