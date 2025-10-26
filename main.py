from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from utils.utils import heading


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def prepare_data(config: dict[str, Any]):
    heading("01 Data Preparation")

    dataset = JapaneseVowelsDataset(
        config=config
        )

def main():
    # Load the config and print current settings
    cfg = load_config()
    heading("Config settings")
    print(yaml.safe_dump(cfg, sort_keys=True))

    # Build augmenter from YAML
    # heading("Augmentation") # Nothing prints here
    aug_cfg = cfg.get("AUGMENTATION", {})
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


if __name__ == "__main__":
    main()
