from sklearn.metrics import accuracy_score, f1_score

from ..core.registry import register_metric


@register_metric("accuracy")
def accuracy(y_true, y_pred, y_proba=None) -> float:
    return float(accuracy_score(y_true, y_pred))


@register_metric("f1_macro")
def f1_macro(y_true, y_pred, y_proba=None) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))
