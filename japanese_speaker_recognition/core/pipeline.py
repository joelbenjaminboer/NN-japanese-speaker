import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy import ndarray

from japanese_speaker_recognition.core.base import BaseModel, BaseTransformer, Split
from japanese_speaker_recognition.core.registry import (
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    TRANSFORMER_REGISTRY,
)


@dataclass
class PipelineArtifacts:
    out_dir: Path
    model_path: Path
    transformers_dir: Path
    meta_path: Path


class Pipeline:
    def __init__(
        self, transformers: list[BaseTransformer], model: BaseModel, out_dir: str | Path
    ) -> None:
        self.transformers: list[BaseTransformer] = transformers
        self.model: BaseModel = model
        self.artifacts: PipelineArtifacts = PipelineArtifacts(
            out_dir=Path(out_dir),
            model_path=Path(out_dir) / "model",
            transformers_dir=Path(out_dir) / "transformers",
            meta_path=Path(out_dir) / "meta.json",
        )

    def _ensure_dirs(self) -> None:
        self.artifacts.out_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.transformers_dir.mkdir(parents=True, exist_ok=True)

    def _apply_transforms_fit(self, x: ndarray, y: ndarray) -> ndarray:
        for i, t in enumerate(self.transformers):
            x = t.fit_transform(x, y)
            # persist each transformer
            (t_path := self.artifacts.transformers_dir / f"{i:02d}_{t.__class__.__name__}.joblib")
            t.save(str(t_path))
        return x

    def _apply_transforms(self, x: ndarray) -> ndarray:
        for t in self.transformers:
            x = t.transform(x)
        return x

    def fit(self, split: Split):
        self._ensure_dirs()
        x_train = self._apply_transforms_fit(split.x_train, split.y_train)
        _ = self.model.fit(x_train, split.y_train)
        self.model.save(str(self.artifacts.model_path))
        self._save_meta()

    def evaluate(self, split: Split, metric_name: str):
        try:
            metric = METRIC_REGISTRY[metric_name]
        except KeyError as exc:
            registered = list(METRIC_REGISTRY.keys())
            raise ValueError(
                f"Unknown metric: {metric_name}. Registered metrics include: {registered}"
            ) from exc
        xv = self._apply_transforms(split.x_valid if split.x_valid is not None else split.x_test)
        yv = split.y_valid if split.y_valid is not None else split.y_test
        return self.model.score(xv, yv, metric)

    def predict(self, x: ndarray) -> ndarray:
        x = self._apply_transforms(x)
        return self.model.predict(x)

    def _save_meta(self):
        meta = {
            "model": getattr(self.model, "__registry_name__", self.model.__class__.__name__),
            "model_params": self.model.get_params(),
            "transformers": [
                getattr(t, "__registry_name__", t.__class__.__name__) for t in self.transformers
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _ = self.artifacts.meta_path.write_text(json.dumps(meta, indent=4))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Pipeline":
        # Build transformers
        transformers: list[BaseTransformer] = []
        for t_cfg in cfg.get("transformers", []):
            name = t_cfg["type"].lower()
            params = t_cfg.get("params", {})
            transformers.append(TRANSFORMER_REGISTRY[name](**params))
        # Build model
        model_name = cfg["model"]["type"].lower()
        model_params = cfg["model"].get("params", {})
        model = MODEL_REGISTRY[model_name](**model_params)
        return cls(transformers=transformers, model=model, out_dir=cfg["training"]["out_dir"])
