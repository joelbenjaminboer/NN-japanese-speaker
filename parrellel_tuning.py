from pathlib import Path

import torch
import yaml

from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from japanese_speaker_recognition.optimization.optuna_tuner import (
    OptunaTuner,  # adapt path if needed
)

# --- Load data and config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

aug_cfg = cfg.get("AUGMENTATION") or {}
augmenter = AugmentationPipeline.from_config(aug_cfg) if aug_cfg else None

embed_cfg = cfg.get("EMBEDDING") or {}
ds = JapaneseVowelsDataset(cfg, 
                            augmenter= augmenter,
                            embedding_dim=embed_cfg.get("DIMENSION", 64),
                            key=embed_cfg.get("KEY", None))
artifacts = ds.prepare()
x_train = artifacts["X_train"]
y_train = artifacts["y_train"]
x_test = artifacts["X_test"]
y_test = artifacts["y_test"]

x_train = torch.tensor(x_train, dtype=torch.float32)  # Shape: [B, 12, 64]
y_train = torch.tensor(y_train, dtype=torch.long)     # Shape: [B, ]    
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

tuner = OptunaTuner(
    x_train=x_train,
    y_train=y_train,
    base_config=cfg,
    n_trials=2,  # total per process, can be smaller since we'll parallelize
    study_name="HAIKU_speaker_recognition",
)

# === Shared storage file ===
storage_url = "sqlite:///habrok_shared/optuna_study.db"
# IMPORTANT: this folder must exist and be on shared storage, e.g., /scratch/<user>/ or $HOME/

Path("habrok_shared").mkdir(exist_ok=True)
tuner.optimize(storage_url=storage_url)
