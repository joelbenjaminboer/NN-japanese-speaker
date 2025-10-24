from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

Array3D = NDArray[np.floating]  # (n_samples, maxlen, n_features)


# ---------- Core Interfaces ----------
class Transform(Protocol):
    def apply(self, x: Array3D, rng: np.random.Generator | None = None) -> Array3D: ...


@dataclass(frozen=True)
class Probabilistic:
    """Optional transform wrapper that applies transform with probability p."""

    transform: Transform
    p: float = 1.0

    def apply(self, x: Array3D, rng: np.random.Generator | None = None) -> Array3D:
        rng = rng or np.random.default_rng()
        if self.p >= 1.0 or rng.random() < self.p:
            return self.transform.apply(x, rng=rng)
        return x


# ---------- Concrete Transforms ----------
@dataclass(frozen=True)
class AddGaussianNoise(Transform):
    noise_factor: float = 0.001

    @override
    def apply(self, x: Array3D, rng: np.random.Generator | None = None) -> Array3D:
        rng = rng or np.random.default_rng()
        noise = rng.normal(0.0, self.noise_factor, size=x.shape)
        return x + noise


@dataclass(frozen=True)
class RandomScaling(Transform):
    scale_range: tuple[float, float] = (0.95, 1.05)

    @override
    def apply(self, x: Array3D, rng: np.random.Generator | None = None) -> Array3D:
        rng = rng or np.random.default_rng()
        scale = rng.uniform(*self.scale_range)
        return x * scale


@dataclass(frozen=True)
class TimeMasking(Transform):
    max_mask_percentage: float = 0.01

    @override
    def apply(self, x: Array3D, rng: np.random.Generator | None = None) -> Array3D:
        rng = rng or np.random.default_rng()
        n, maxlen, f = x.shape
        mask_len = max(1, int(maxlen * self.max_mask_percentage))
        Y = x.copy()
        if mask_len >= maxlen:
            return Y
        starts = rng.integers(0, maxlen - mask_len, size=n)
        for i, s in enumerate(starts):
            Y[i, s : s + mask_len, :] = 0.0
        return Y


@dataclass(frozen=True)
class FrequencyMasking(Transform):
    max_mask_percentage: float = 0.01

    @override
    def apply(self, x: Array3D, rng: np.random.Generator | None = None) -> Array3D:
        rng = rng or np.random.default_rng()
        n, maxlen, f = x.shape
        mask_len = max(1, int(f * self.max_mask_percentage))
        Y = x.copy()
        if mask_len >= f:
            return Y
        starts = rng.integers(0, f - mask_len, size=n)
        for i, s in enumerate(starts):
            Y[i, :, s : s + mask_len] = 0.0
        return Y


# ---------- Pipeline ----------
@dataclass
class AugmentationPipeline:
    steps: list[Probabilistic]
    seed: int | None = None

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def apply_once(self, x: Array3D) -> Array3D:
        y = x
        rng = self._rng()
        for step in self.steps:
            y = step.apply(y, rng=rng)
        return y

    def run(self, X: Array3D, repeats: int) -> tuple[Array3D, NDArray[np.int_]]:
        """
        Creates `repeats` augmented copies of X (same size as X per repeat),
        then stacks them vertically (axis=0). Returns (X_aug, idx_map) where idx_map
        is an index array mapping each augmented sample to its original sample id.
        """
        if repeats <= 0:
            return np.empty((0, *X.shape[1:]), dtype=X.dtype), np.empty((0,), dtype=int)

        batches = []
        for _ in range(repeats):
            batches.append(self.apply_once(X))
        X_aug = np.vstack(batches)  # (repeats * n, maxlen, n_features)

        # Map: for each batch we tile range(n)
        n = X.shape[0]
        idx_map = np.tile(np.arange(n, dtype=int), repeats)
        return X_aug, idx_map

    # ---------- Factory from config ----------
    @staticmethod
    def from_config(cfg: dict[str, Any]) -> AugmentationPipeline:
        steps_cfg = (cfg or {}).get("STEPS", []) or []
        seed = (cfg or {}).get("SEED", None)

        type_map = {
            "gaussian_noise": AddGaussianNoise,
            "random_scaling": RandomScaling,
            "time_masking": TimeMasking,
            "frequency_masking": FrequencyMasking,
        }

        steps: list[Probabilistic] = []
        for sc in steps_cfg:
            type = (sc.get("type") or "").lower().strip()
            p = float(sc.get("p", 1.0))
            if type not in type_map:
                raise ValueError(f"Unknown augmentation type: {type}")
            params = {k: v for k, v in sc.items() if k not in {"type", "p"}}
            transform = type_map[type](**params)  # dataclass init
            steps.append(Probabilistic(transform=transform, p=p))

        return AugmentationPipeline(steps=steps, seed=seed)
