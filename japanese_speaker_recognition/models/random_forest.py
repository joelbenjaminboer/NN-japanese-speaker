"""
random_forest_model.py
---------------------------------
Wrapper class around scikit-learn's RandomForestClassifier
for the Japanese Vowels dataset.
"""

import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing_extensions import override

from ..core.base import BaseModel
from ..core.registry import register_model


@register_model("random_forest")
class RandomForestWrapper(BaseModel):
    name = "random_forest"

    def __init__(self, n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, **kwargs):
        """
        Initialize a RandomForest classifier with desired hyperparameters.
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.clf = RandomForestClassifier(**self.params)

    @staticmethod
    def load_npz(train_file, test_file):
        train = np.load(train_file)
        test = np.load(test_file)

        # Handle either normal or augmented naming
        if "X_train" in train:
            X_train, y_train = train["X_train"], train["y_train"]
        elif "X_augmented" in train:
            X_train, y_train = train["X_augmented"], train["y_augmented"]
        else:
            raise KeyError("No valid X/y keys found in training file")

        X_test, y_test = test["X_test"], test["y_test"]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def flatten(x: ndarray) -> ndarray:
        """
        Flatten a (samples, time, features) array to (samples, time*features)
        for use in classical ML models.
        """
        return x.reshape(x.shape[0], -1)

    @override
    def fit(self, x: ndarray, y: ndarray):
        """
        Train the RandomForest model.
        """
        print("Training Random Forest...")
        self.clf.fit(x, y)
        self._is_fitted = True
        return self

    def predict(self, x: ndarray):
        """
        Predict labels for input samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained yet. Call `.fit()` first.")
        return self.clf.predict(x)

    def predict_proba(self, x: ndarray):
        if not self._is_fitted:
            raise RuntimeError("Model not trained yet. Call `.fit()` first.")
        return self.clf.predict_proba(x)

    def evaluate(self, X_test, y_test):
        """
        DEPRECATED METHOD
        Use model.score() instead with a metric chosen from the metric registry.
        Evaluate accuracy and return classification report.
        """
        if np.all(y_test == -1):
            print("Test labels unknown")
            return None

        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.3f}")
        print("\nDetailed report:")
        print(classification_report(y_test, y_pred))
        return acc
