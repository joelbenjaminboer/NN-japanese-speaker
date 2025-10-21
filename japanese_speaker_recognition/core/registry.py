# Registry for models, transformers, and metrics.
MODEL_REGISTRY: dict[str, type] = {}
TRANSFORMER_REGISTRY: dict[str, type] = {}
METRIC_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator method to register a model by unique name."""

    def _decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        cls._model_name = name
        MODEL_REGISTRY[name] = cls
        cls.__registry_name__ = name
        return cls

    return _decorator


def register_transformer(name: str):
    """Decorator method to register a transformer by unique name."""

    def _decorator(cls):
        if name in TRANSFORMER_REGISTRY:
            raise ValueError(f"Transformer '{name}' is already registered.")
        cls._transformer_name = name
        TRANSFORMER_REGISTRY[name] = cls
        cls.__registry_name__ = name
        return cls

    return _decorator


def register_metric(name: str):
    """Decorator method to register a metric by unique name."""

    def _decorator(cls):
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        cls._metric_name = name
        METRIC_REGISTRY[name] = cls
        cls.__registry_name__ = name
        return cls

    return _decorator
