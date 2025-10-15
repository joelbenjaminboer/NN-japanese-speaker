"""
random_forest_model.py
---------------------------------
Wrapper class around scikit-learn's RandomForestClassifier
for the Japanese Vowels dataset.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


class RandomForestWrapper:
    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        """
        Initialize a RandomForest classifier with desired hyperparameters.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
        self.is_trained = False

    @staticmethod
    def load_npz(train_path: str, test_path: str):
        """
        Load preprocessed train and test .npz files.
        Returns X_train, y_train, X_test, y_test.
        """
        train = np.load(train_path)
        test = np.load(test_path)
        X_train, y_train = train["X_train"], train["y_train"]
        X_test, y_test = test["X_test"], test["y_test"]

        print(f"Loaded data:\n"
              f"  Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def flatten(X):
        """
        Flatten a (samples, time, features) array to (samples, time*features)
        for use in classical ML models.
        """
        return X.reshape(X.shape[0], -1)

    def train(self, X_train, y_train):
        """
        Train the RandomForest model.
        """
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")

    def predict(self, X):
        """
        Predict labels for input samples.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call `.train()` first.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
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