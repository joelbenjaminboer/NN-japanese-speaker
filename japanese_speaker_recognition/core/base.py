from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Self

import joblib
from numpy import ndarray


class SupportsProba(Protocol):
    def predict_proba(self, x) -> float: ...


class BaseTransformer(ABC):
    @abstractmethod
    def fit(self, x: ndarray, y: ndarray | None = None) -> Self:
        return self

    @abstractmethod
    def transform(self, x: ndarray) -> ndarray: ...

    def fit_transform(self, x: ndarray, y: ndarray | None = None) -> ndarray:
        _ = self.fit(x, y)
        return self.transform(x)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


class BaseModel(ABC):
    """Minimal interface for a machine learning model."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.params: dict[str, Any] = kwargs  # pyright: ignore[reportExplicitAny]
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, x: ndarray, y: ndarray) -> "BaseModel":
        """Train and return self"""
        ...

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray: ...

    def predict_proba(self, x: ndarray):  # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError

    def score(self, x: ndarray, y: ndarray, metric):
        """Compute score using a callable metric(y_true, y_pred[, y_proba])."""
        if hasattr(self, "predict_proba"):
            try:
                proba = self.predict_proba(x)
                y_pred = self.predict(x)
                return metric(y, y_pred, proba)
            except NotImplementedError:
                pass
        y_pred = self.predict(x)
        return metric(y, y_pred)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)

    def get_params(self) -> dict[str, Any]:
        return self.params

    def set_params(self, **params) -> Self:
        self.params.update(params)
        return self


@dataclass
class Split:
    x_train: ndarray
    x_test: ndarray
    x_valid: ndarray
    y_valid: ndarray
    y_train: ndarray
    y_test: ndarray
