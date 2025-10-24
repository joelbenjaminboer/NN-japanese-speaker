from typing import Self

from numpy import ndarray
from typing_extensions import override

from ..core.base import BaseTransformer
from ..core.registry import register_transformer


@register_transformer("flatten")
class Flatten(BaseTransformer):
    """Flattens (n_samples, time, channels) → (n_samples, time*channels)."""

    @override
    def fit(self, x: ndarray, y: ndarray | None = None) -> Self:
        return self

    @override
    def transform(self, x: ndarray) -> ndarray:
        return x.reshape(x.shape[0], -1)
