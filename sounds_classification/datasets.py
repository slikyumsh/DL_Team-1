import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from scipy.ndimage import gaussian_filter
from dotenv import load_dotenv
from tqdm import tqdm
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s"
)

model_config = "./model.env"
load_dotenv(model_config)
load_dotenv(os.getenv("PREPROCESS_PATH"))

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, use_augmented=False):
        super().__init__()
        self.preproc_dir = os.path.join(os.getenv("DATA_ROOT"), "processed")
        self.aug_dir = os.path.join(os.getenv("DATA_ROOT"), "processed_aug")
        self.data_dir = os.path.join(self.preproc_dir, root_dir)
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Оригинальные данные
        for label in self.classes:
            class_dir = os.path.join(self.data_dir, label)
            for fname in os.listdir(class_dir):
                self.samples.append(os.path.join(class_dir, fname))
                self.labels.append(self.class_to_idx[label])

        # Аугментированные
        if use_augmented:
            aug_subset_dir = os.path.join(self.aug_dir, root_dir)
            if os.path.exists(aug_subset_dir):
                for label in os.listdir(aug_subset_dir):
                    aug_class_dir = os.path.join(aug_subset_dir, label)
                    for fname in os.listdir(aug_class_dir):
                        self.samples.append(os.path.join(aug_class_dir, fname))
                        self.labels.append(self.class_to_idx[label])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        spectrogram = np.load(self.samples[idx])
        if spectrogram.ndim == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Добавить канал: (1, H, W)
        label = self.labels[idx]
        return torch.tensor(spectrogram, dtype=torch.float32), label # TODO обратно к float16
    
    def generate_augmented_data(self, subset="train", augmentations_per_sample=2):
        """
        Генерирует аугментированные спектрограммы для указанного подмножества (по умолчанию: train).
        """
        logging.info(f"Генерация аугментированных данных для '{subset}'...")

        input_dir = os.path.join(self.preproc_dir, subset)
        output_dir = os.path.join(self.aug_dir, subset)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for label in tqdm(os.listdir(input_dir)):
            label_input_dir = os.path.join(input_dir, label)
            label_output_dir = os.path.join(output_dir, label)
            os.makedirs(label_output_dir, exist_ok=True)

            for fname in os.listdir(label_input_dir):
                spec = np.load(os.path.join(label_input_dir, fname))

                for i in range(augmentations_per_sample):
                    augmented = self.augment_spectrogram(spec)
                    aug_name = fname.replace(".npy", f"_aug{i}.npy")
                    np.save(os.path.join(label_output_dir, aug_name), augmented)

        logging.info(f"Аугментированные данные сохранены в: {output_dir}")

    @staticmethod
    def augment_spectrogram(spec):
        # Добавить шум
        noise = np.random.normal(0, 0.02, spec.shape)
        spec_noisy = spec + noise

        # Гауссовое размытие с вероятностью
        if random.random() < 0.5:
            spec_noisy = gaussian_filter(spec_noisy, sigma=0.5)

        # Time mask
        if random.random() < 0.3:
            t = random.randint(0, spec.shape[1] - 10)
            width = random.randint(5, 15)
            spec_noisy[:, t:t+width] = 0

        # Frequency mask
        if random.random() < 0.3:
            f = random.randint(0, spec.shape[0] - 5)
            height = random.randint(5, 10)
            spec_noisy[f:f+height, :] = 0

        # Нормализация
        spec_noisy = np.clip(spec_noisy, 0, 1)
        return spec_noisy