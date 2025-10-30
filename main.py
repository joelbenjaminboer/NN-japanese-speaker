from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import TensorDataset

from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from japanese_speaker_recognition.models.HAIKU import HAIKU
from japanese_speaker_recognition.optimization.optuna_tuner import OptunaTuner
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

def plot_training_history(history: dict, figure_dir: Path) -> None:
    """Plot training history."""
    heading("Training History")
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "training_history.png")

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
    x_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    
    

    x_train = torch.tensor(x_train, dtype=torch.float32)  # Shape: [B, 12, 64]
    y_train = torch.tensor(y_train, dtype=torch.long)     # Shape: [B, ]    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # quick summary (shapes + file outputs)
    heading("Artifacts")

    figures_dir = Path(
        cfg.get("OPTUNA", {}).get("FIGURES_DIR", "reports/optuna/figures")
        )
    study_dir = Path(
        cfg.get("OPTUNA", {}).get("STUDY_DIR", "reports/optuna/study")
        )
    best_config_dir = Path(
        cfg.get("OPTUNA", {}).get("BEST_CONFIG_DIR", "reports/optuna/best_config")
        )
    figures_dir.mkdir(parents=True, exist_ok=True)
    best_config_dir.mkdir(parents=True, exist_ok=True)
    study_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("OPTUNA", {}).get("ENABLED", False):
        tuner = OptunaTuner(
            x_train=x_train,
            y_train=y_train,
            base_config=cfg,
            n_trials=cfg.get("OPTUNA", {}).get("N_TRIALS", 50),
            study_name=cfg.get("OPTUNA", {}).get("STUDY_NAME", "HAIKU_speaker_recognition"),
            seed=cfg.get("SEED", 42),
        )

        _ = tuner.optimize()

        figures_dir.mkdir(parents=True, exist_ok=True)
        study_dir.mkdir(parents=True, exist_ok=True)
        best_config_dir.mkdir(parents=True, exist_ok=True)

        tuner.save_plots(output_dir=figures_dir)
        tuner.save_study(output_dir=study_dir / "optuna_study.pkl")

        model_cfg = tuner.get_best_config()
        # save the config
        with open(best_config_dir / "best_model_config.yaml", "w") as f:
            yaml.dump(model_cfg, f)

    heading("Model Creation")

    if cfg.get("MODEL", {}).get("LOAD_BEST_CONFIG", False):
        with open(best_config_dir / "best_model_config.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
            print("Loaded best model config:")
            print(yaml.safe_dump(model_cfg, sort_keys=True, default_flow_style=False))
    else:
        model_cfg = cfg.get("MODEL", {})
    
    model = HAIKU.create_model(model_cfg)
    
    # Print detailed model summary
    batch_size = model_cfg.get("BATCH_SIZE", 32)

    # Train the model
    history, avg_history = model.train_model(
        x_train,
        y_train,
        learning_rate = model_cfg.get("LEARNING_RATE", 0.001),
        num_epochs = model_cfg.get("NUM_EPOCHS", 500),  
        batch_size = batch_size,
        k_folds= model_cfg.get("K_FOLDS", 5),
        seed= cfg.get("SEED", 42)
    )
    
    figure_dir = Path("reports/figures")
    # plot the training history
    plot_training_history(history, figure_dir)

    # Print final results
    heading("Training Complete")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    
    # evaluate on test set
    test_set = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    avg_loss, test_acc = model.evaluate(test_loader, criterion=nn.CrossEntropyLoss())
    print(f"Test Set Accuracy: {test_acc:.2f}%")
    print(f"Test Set Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
