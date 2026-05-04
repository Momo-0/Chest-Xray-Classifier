import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy
import h5py
import os


# ──────────────────────────────────────────────
#  Data Transforms
# ──────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ──────────────────────────────────────────────
#  Dataset & DataLoaders
# ──────────────────────────────────────────────

def get_dataloaders(data_dir: str, batch_size: int = 32):
    """
    Expects the following folder structure:
        data_dir/
            train/
                COVID/
                Normal/
                Pneumonia/
            val/          (optional — currently mirrors train)
                ...
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f"Classes detected: {train_dataset.classes}")
    return train_loader, val_loader, train_dataset.classes


# ──────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────

def build_model(num_classes: int, device: torch.device):
    """
    Load a pretrained DenseNet-161 and replace the classifier
    head with a linear layer matching num_classes.
    """
    model = models.densenet161(pretrained=True)

    # Freeze backbone weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model.to(device)


# ──────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────

def train_model(model, criterion, optimizer,
                train_loader, val_loader,
                device, num_epochs: int = 25):
    """
    Train the model and return the best weights plus
    per-epoch loss / accuracy history.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_losses':     [],
        'val_losses':       [],
        'train_accuracies': [],
        'val_accuracies':   [],
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        for phase, loader in [('train', train_loader), ('val', val_loader)]:
            model.train() if phase == 'train' else model.eval()

            running_loss     = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc  = running_corrects.double() / len(loader.dataset)

            history[f'{phase}_losses'].append(epoch_loss)
            history[f'{phase}_accuracies'].append(epoch_acc.item())

            print(f'  {phase.capitalize():5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\nTraining complete.  Best val accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history


# ──────────────────────────────────────────────
#  Save Artifacts
# ──────────────────────────────────────────────

def save_results(model, history, model_dir: str = 'saved_models'):
    os.makedirs(model_dir, exist_ok=True)

    # Model weights
    torch.save(model.state_dict(),
               os.path.join(model_dir, 'best_model_weights.pth'))

    # Full model (architecture + weights)
    torch.save(model,
               os.path.join(model_dir, 'best_model.pth'))

    # Training metrics
    with h5py.File('results/training_results.h5', 'w') as f:
        for key, values in history.items():
            f.create_dataset(key, data=np.array(values))

    print(f"Model saved to '{model_dir}/'")
    print("Metrics saved to 'results/training_results.h5'")


# ──────────────────────────────────────────────
#  Plot Metrics
# ──────────────────────────────────────────────

def plot_history(history, save_path: str = 'results/training_curves.png'):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(history['train_losses'], label='Train Loss')
    axs[0].plot(history['val_losses'],   label='Val Loss')
    axs[0].set_title('Training vs Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(history['train_accuracies'], label='Train Accuracy')
    axs[1].plot(history['val_accuracies'],   label='Val Accuracy')
    axs[1].set_title('Training vs Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to '{save_path}'")


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR   = '/home/ahmed/Downloads'   # ← change to your dataset path
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LR         = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    model     = build_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    model, history = train_model(
        model, criterion, optimizer,
        train_loader, val_loader,
        device, num_epochs=NUM_EPOCHS
    )

    save_results(model, history)
    plot_history(history)
