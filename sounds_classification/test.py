from datasets import SpectrogramDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s"
)

def sanity_check_loader(loader, num_classes, expected_shape):
    """
    loader           — DataLoader, который нужно проверить
    num_classes      — число классов (len(class_names))
    expected_shape   — кортеж (C, H, W), например (1, 128, 128)
    """
    # 1) Забираем первую батч‑выборку
    specs, labels = next(iter(loader))
    
    # 2) Shape и типы
    assert isinstance(specs, torch.Tensor), "Спектрограммы не Tensor"
    assert isinstance(labels, torch.Tensor), "Лейблы не Tensor"
    b, c, h, w = specs.shape
    assert c == expected_shape[0] and h == expected_shape[1] and w == expected_shape[2], \
        f"Неправильный размер спектрограммы: ожидается {expected_shape}, получили {specs.shape[1:]}"
    assert labels.dtype == torch.int64, "Лейблы должны быть типа LongTensor"
    assert labels.min().item() >= 0 and labels.max().item() < num_classes, \
        f"Лейблы выходят за диапазон [0, {num_classes-1}]"
    logging.info(f"[OK] Батч: {b} образцов, спектрограммы {c}×{h}×{w}, метки в диапазоне 0–{num_classes-1}")
    
    # 3) Распределение меток в батче
    unique, counts = torch.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print("Распределение меток в первом батче:", dist)

    # 4) Визуализация первых 6 спектрограмм
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes.flatten()):
        spec = specs[i, 0].cpu().numpy()
        ax.imshow(spec, origin='lower', aspect='auto')
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    logging.info("creating datasets")
    train_dataset = SpectrogramDataset("train")
    val_dataset = SpectrogramDataset("val")
    test_dataset = SpectrogramDataset("test")

    logging.info("creating dataloaders")
    batch_size = int(os.getenv("BATCH_SIZE"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logging.info("testing dataloaders")
    class_names = train_dataset.classes
    shape = tuple(int(num) for num in os.getenv("SHAPE").split(','))
    sanity_check_loader(train_loader, num_classes=len(class_names), expected_shape=shape)
    sanity_check_loader(val_loader,   num_classes=len(class_names), expected_shape=shape)
    sanity_check_loader(test_loader,  num_classes=len(class_names), expected_shape=shape)