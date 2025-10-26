from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import torch

from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from japanese_speaker_recognition.models.cnn import SpeakerCNN
from utils.utils import heading


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    
    # ========== Simple CNN Model Usage (Original) ==========
    
    heading("Simple CNN Model - Basic Usage")
    
    model = SpeakerCNN(num_classes=9, dropout=0.3)
    print(f"Model architecture:")
    print(f"  - Input: [Batch, 12 features, 64 embedding_dim]")
    print(f"  - Conv layers: 12→32→64→128 channels")
    print(f"  - Kernel sizes: [3, 5, 5]")
    print(f"  - Pooling: MaxPool after 2nd and 3rd conv")
    print(f"  - Output: [Batch, 9 classes]")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example: Forward pass with dummy data
    heading("Forward Pass Example")
    batch_size = 8
    dummy_input = torch.randn(batch_size, 12, 64)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].tolist()}")
    
    # Example: Extract features (without classification)
    heading("Feature Extraction Example")
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Features are 128-dim representations for each sample")
    
    # Example: Training mode
    heading("Training Setup Example")
    model.train()
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy training step
    dummy_labels = torch.randint(0, 9, (batch_size,))
    
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()
    
    print(f"Loss function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Sample loss: {loss.item():.4f}")
    
    # Example: Model summary
    heading("Model Architecture Details")
    print(model)


if __name__ == "__main__":
    main()
