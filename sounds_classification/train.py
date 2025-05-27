import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from datasets import SpectrogramDataset
from model import SpectrogramClassifier, SpectrogramClassifierResNet, ImprovedSpectrogramClassifier
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s"
)

def train_model(model: torch.nn.Module,
                train_loader: DataLoader, 
                val_loader: DataLoader,  
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                criterion, 
                num_epochs=10):
    
    model.to(device)
    scaler = GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):  # Автоматический FP16
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct/total)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        logging.info(f"Epoch {epoch + 1}: Train Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast("cuda"):  # Автоматический FP16
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        logging.info(f"Validation Accuracy: {val_acc:.4f}\n")

def evaluate_model(model: torch.nn.Module, 
                   loader: DataLoader, 
                   device: torch.device, 
                   phase: str = "Test") -> None:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast("cuda"):  # Автоматический FP16
                outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    logging.info(f"\n--- {phase} Evaluation ---")
    logging.info(classification_report(all_labels, all_preds, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=90, cmap='Blues')
    plt.title(f"{phase} Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":

    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    EPOCHS = int(os.getenv("EPOCHS"))
    LR = float(os.getenv("LR")) # 1e-3
    NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
    AUG_P_SAMP = int(os.getenv("AUG_P_SAMP"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp_dataset = SpectrogramDataset("train")
    temp_dataset.generate_augmented_data(subset="train", augmentations_per_sample=AUG_P_SAMP)
    train_dataset = SpectrogramDataset("train", use_augmented=True)

    val_dataset = SpectrogramDataset("val")
    test_dataset = SpectrogramDataset("test")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ImprovedSpectrogramClassifier(num_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_model(model,
                train_dataloader,
                val_loader,
                optimizer,
                device,
                criterion,
                num_epochs=EPOCHS)
    
    evaluate_model(model, test_loader, device, "Test")