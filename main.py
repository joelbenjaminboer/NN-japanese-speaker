from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

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
    x_train = artifacts["X_train"]
    y_train = artifacts["y_train"]
    x_val = artifacts["X_test"]
    y_val = artifacts["y_test"]

    # quick summary (shapes + file outputs)
    heading("Artifacts")
    for k, v in artifacts.items():
        if hasattr(v, "shape"):
            print(f"{k:15s} -> shape={v.shape}")
        else:
            print(f"{k:15s} -> {v}")

    # Get model config and create model
    heading("Model Creation")
    model_cfg = cfg.get("MODEL", {})
    model = HAIKU.create_model(model_cfg)
    
    # Print detailed model summary
    batch_size = model_cfg.get("BATCH_SIZE", 32)
    embedding_dim = model_cfg.get("EMBEDDING_DIM", 64)
    input_channels = model_cfg.get("INPUT_CHANNELS", 12)

    print_model_summary(model, (batch_size, input_channels, embedding_dim))

    # Train the model
    history = model.train_model(
        x_train,
        y_train,
        x_val,
        y_val,
        learning_rate=model_cfg.get("LEARNING_RATE", 0.007),
        num_epochs=model_cfg.get("NUM_EPOCHS", 10),
        batch_size=model_cfg.get("BATCH_SIZE", 32)
    )

    # Print final results
    heading("Training Complete")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")


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



if __name__ == "__main__":
    main()
