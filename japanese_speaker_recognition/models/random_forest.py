"""
random_forest_model.py
---------------------------------
Wrapper class around scikit-learn's RandomForestClassifier
for the Japanese Vowels dataset.
"""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing_extensions import override


class RandomForestWrapper(RandomForestClassifier):

    def __init__(
        self, 
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
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
    def flatten(x:  NDArray[np.float_]) ->  NDArray[np.float_]:
        """
        Flatten a (samples, time, features) array to (samples, time*features)
        for use in classical ML models.
        """
        return x.reshape(x.shape[0], -1)

    @override
    def fit(
        self, 
        X: NDArray[np.float_], 
        y: NDArray[np.int_], 
        sample_weight: NDArray[np.float_] | None = None
        ) -> Self:
        """
        Train the RandomForest model.
        """
        print(f"Training Random Forest with {getattr(self, 'n_estimators', 'unknown')} \
            estimators...")
        super().fit(X, y, sample_weight=sample_weight)
        print("Training complete!")
        return self

    def evaluate(self, X_test, y_test):
        """
        DEPRECATED METHOD
        Use model.score() instead with a metric chosen from the metric registry.
        Evaluate accuracy and return classification report.
        """
        if np.all(y_test == -1):
            print("Test labels unknown")
            return 0.0

        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return acc
