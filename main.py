from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import TensorDataset

from config.config import Config, Model
from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from japanese_speaker_recognition.models.HAIKU import HAIKU
from japanese_speaker_recognition.optimization.optuna_tuner import OptunaTuner
from utils.utils import heading


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
    # -------------------------------------------
    # Load the config
    # -------------------------------------------
    heading("Config settings")
    cfg = Config.from_yaml()
    print(cfg)

    # -------------------------------------------
    # Device Configuration
    # -------------------------------------------
    heading("Device Configuration")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.model.device != device:
        print(f"Warning: Configured device {cfg.model.device} is not available. \
            Using {device} instead.")
    else:
        print(f"Using device: {device}")

    # -------------------------------------------
    # Build data augmentation pipeline from YAML
    # -------------------------------------------
    augmenter = AugmentationPipeline.from_config(cfg)

    # -------------------------------------------
    # Prep the dataset
    # -------------------------------------------
    heading("Preparing dataset")
    embed_cfg = cfg.embedding
    ds = JapaneseVowelsDataset(cfg, 
                               augmenter=augmenter,
                               embedding_dim=embed_cfg.dimension,
                               key=embed_cfg.key)

    artifacts = ds.prepare()
    x_train = artifacts["X_train"]
    y_train = artifacts["y_train"]
    x_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    x_train = torch.tensor(x_train, dtype=torch.float32)  # Shape: [B, 12, 64]
    y_train = torch.tensor(y_train, dtype=torch.long)     # Shape: [B, ]    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Move tensors to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

    # -------------------------------------------
    # Optuna Hyperparameter Tuning
    # -------------------------------------------
    # TODO: save the best model as a pkl for easy retrieval.
    figures_dir = cfg.optuna.figures_dir
    study_dir = cfg.optuna.study_dir
    best_config_dir = cfg.optuna.best_config_dir

    figures_dir.mkdir(parents=True, exist_ok=True)
    study_dir.mkdir(parents=True, exist_ok=True)
    best_config_dir.mkdir(parents=True, exist_ok=True)

    if cfg.optuna.enabled:
        tuner = OptunaTuner(
            x_train=x_train,
            y_train=y_train,
            config=cfg,
            n_trials=cfg.optuna.n_trials,
            study_name=cfg.optuna.study_name,
            seed=cfg.seed,
            device=device
        )

        _ = tuner.optimize(show_progress_bar=cfg.optuna.show_progress_bar)

        tuner.save_plots(output_dir=figures_dir)
        tuner.save_study(output_dir=study_dir / "optuna_study.pkl")

        model_cfg = tuner.get_best_config()
        # save the config
        with open(best_config_dir / "best_model_config.yaml", "w") as f:
            yaml.dump(model_cfg, f)

    # -------------------------------------------
    # Model Creation from config
    # -------------------------------------------
    heading("Model Creation")

    if cfg.model.load_best_config:
        with open(best_config_dir / "best_model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)
            model_cfg = Model.from_dict(model_cfg)
            print("Loaded best model config:")
            print(model_cfg)
    else:
        model_cfg = cfg.model
    
    model = HAIKU.create_model(model_cfg).to(device)
    print(f"Model created and moved to device: {device}")

    batch_size = model_cfg.batch_size

    # Train the model
    history, _avg_history = model.train_model(
        x_train,
        y_train,
        learning_rate = model_cfg.learning_rate,
        num_epochs = model_cfg.num_epochs,
        batch_size = batch_size,
        k_folds= model_cfg.k_folds,
        seed= cfg.seed
    )

    # plot the training history
    plot_training_history(history, cfg.output_dirs.figures_dir)

    # Print final results
    heading("Training Complete")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")

    # -------------------------------------------
    # Model Evaluation
    # -------------------------------------------
    test_set = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    avg_loss, test_acc = model.evaluate(test_loader, criterion=nn.CrossEntropyLoss())
    print(f"Test Set Accuracy: {test_acc:.2f}%")
    print(f"Test Set Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
