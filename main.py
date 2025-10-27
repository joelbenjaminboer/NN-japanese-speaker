from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from japanese_speaker_recognition.models.cnn import HAIKU
from utils.utils import heading


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_device(device_config: str = "auto") -> str:
    """
    Get the device to use for training.

    Args:
        device_config: Device specification from config ("cuda", "cpu", or "auto")

    Returns:
        Device string ("cuda" or "cpu")
    """
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_config == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"

    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")

    return device


def create_model(model_cfg: dict) -> HAIKU:
    """Create CNN model from configuration."""
    model = HAIKU.from_config(model_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model created:")
    print(
        f"  - Input: [Batch, {model_cfg.get('INPUT_CHANNELS', 12)}, \
        {model_cfg.get('EMBEDDING_DIM', 64)}]"
    )
    print(f"  - Conv channels: {model_cfg.get('CONV_CHANNELS', 128)}")
    print(f"  - Kernel size: {model_cfg.get('KERNEL_SIZE', 3)}")
    print(f"  - Hidden dim: {model_cfg.get('HIDDEN_DIM', 64)}")
    print(f"  - Output: [Batch, {model_cfg.get('NUM_CLASSES', 9)}]")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    return model


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> tuple[float, float]:
    """Perform one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str = "cpu"
) -> tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_cfg: dict,
    device: str = "cpu",
) -> dict:
    """Train the model and return training history."""
    learning_rate = model_cfg.get("LEARNING_RATE", 0.007)
    num_epochs = model_cfg.get("NUM_EPOCHS", 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    model = model.to(device)

    heading(f"Training for {num_epochs} epochs (lr={learning_rate}, device={device})")

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    return history


def print_model_summary(model: nn.Module, input_shape: tuple):
    """Print detailed model architecture."""
    heading("Model Architecture")
    print(model)

    heading("Layer-by-layer output shapes")
    dummy_input = torch.randn(*input_shape)
    model.eval()

    with torch.no_grad():
        x = dummy_input
        print(f"Input: {x.shape}")

        x = model.conv(x)
        print(f"After Conv: {x.shape}")

        x = model.global_pool(x)
        print(f"After Global Pool: {x.shape}")

        logits = model.classifier(x)
        print(f"After Classifier (MLP): {logits.shape}")


def main():
    # Load the config and print current settings
    cfg = load_config()
    heading("Config settings")
    print(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False))

    # Build augmenter from YAML
    # heading("Augmentation") # Nothing prints here
    aug_cfg = cfg.get("AUGMENTATION") or {}
    augmenter = AugmentationPipeline.from_config(aug_cfg) if aug_cfg else None

    # Prep the dataset
    heading("Preparing dataset")
    ds = JapaneseVowelsDataset(cfg, augmenter=augmenter)
    artifacts = ds.prepare()

    # quick summary (shapes + file outputs)
    heading("Artifacts")
    for k, v in artifacts.items():
        if hasattr(v, "shape"):
            print(f"{k:15s} -> shape={v.shape}")
        else:
            print(f"{k:15s} -> {v}")

    # Get model config
    model_cfg = cfg.get("MODEL", {})

    # Get device from config
    heading("Device Configuration")
    device_config = model_cfg.get("DEVICE", "auto")
    device = get_device(device_config)

    # Create model
    heading("Model Creation")
    model = create_model(model_cfg)

    # Print detailed model summary
    batch_size = model_cfg.get("BATCH_SIZE", 32)
    embedding_dim = model_cfg.get("EMBEDDING_DIM", 64)
    input_channels = model_cfg.get("INPUT_CHANNELS", 12)

    print_model_summary(model, (batch_size, input_channels, embedding_dim))

    # # Create dummy data for demonstration
    # heading("Creating Dummy DataLoaders")
    # num_train_samples = 200
    # num_val_samples = 50
    # num_classes = model_cfg.get("NUM_CLASSES", 9)

    # train_x = torch.randn(num_train_samples, input_channels, embedding_dim)
    # train_y = torch.randint(0, num_classes, (num_train_samples,))
    # val_x = torch.randn(num_val_samples, input_channels, embedding_dim)
    # val_y = torch.randint(0, num_classes, (num_val_samples,))

    # train_dataset = TensorDataset(train_x, train_y)
    # val_dataset = TensorDataset(val_x, val_y)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # print(f"Train samples: {num_train_samples}, Val samples: {num_val_samples}")
    # print(f"Batch size: {batch_size}")
    # print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # # Train the model
    # history = train_model(model, train_loader, val_loader, model_cfg, device)

    # # Print final results
    # heading("Training Complete")
    # print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    # print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    # print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")


if __name__ == "__main__":
    main()
