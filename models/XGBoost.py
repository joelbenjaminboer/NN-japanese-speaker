"""
xgboost_model.py
---------------------------------
Wrapper class around XGBoost's XGBClassifier
for the Japanese Vowels dataset.
"""

import numpy as np
from pathlib import Path

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Install it with:\n\n"
        "  uv add xgboost  # or: pip install xgboost\n"
    ) from e

from sklearn.metrics import accuracy_score, classification_report


class XGBoostWrapper:
    def __init__(
        self,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",      # for >=2.0 use "hist" + device='cuda'
        device="cuda",           # 'cuda' or 'cpu'
        verbosity=1,
        max_bin=256,
    ):
        """
        Initialize an XGBoost classifier with sensible defaults.
        Objective/num_class are set at train time based on label cardinality.
        """
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method=tree_method,
            verbosity=verbosity,
            max_bin=max_bin,
            # IMPORTANT: include device so it reaches the classifier
            device=device,
            predictor="auto",
        )
        self.model = None
        self.is_trained = False

    @staticmethod
    def load_npz(train_file, test_file):
        train = np.load(train_file)
        test = np.load(test_file)

        if "X_train" in train:
            X_train, y_train = train["X_train"], train["y_train"]
        elif "X_augmented" in train:
            X_train, y_train = train["X_augmented"], train["y_augmented"]
        else:
            raise KeyError("No valid X/y keys found in training file")

        X_test, y_test = test["X_test"], test["y_test"]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def flatten(X):
        return X.reshape(X.shape[0], -1)

    def _build_model_for_labels(self, y):
        """Choose objective/num_class/eval_metric based on label cardinality."""
        classes = np.unique(y)
        n_classes = len(classes)

        params = self.params.copy()

        # Back-compat for xgboost < 2.0 (no 'device' param):
        if "device" in params:
            try:
                # touch to ensure the param is supported; otherwise fall back
                _ = XGBClassifier(**{**params, "objective": "binary:logistic"})
            except TypeError:
                # remove device, use gpu_hist if we asked for cuda
                dev = params.pop("device", "cpu")
                if dev == "cuda":
                    params["tree_method"] = "gpu_hist"
                    params["predictor"] = "gpu_predictor"

        if n_classes <= 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softmax", num_class=n_classes, eval_metric="mlogloss")

        self.model = XGBClassifier(**params)

    def train(self, X_train, y_train):
        if X_train.ndim > 2:
            X_train = self.flatten(X_train)
        y_train = y_train.ravel()

        print(f"Configuring XGBoost (xgboost {xgb.__version__})...")
        self._build_model_for_labels(y_train)

        # Print where it will run
        xgb_params = self.model.get_xgb_params()
        dev = xgb_params.get("device", "(no 'device' param; using predictor/tree_method)")
        print(f"Using device: {dev}, tree_method: {xgb_params.get('tree_method')}, predictor: {xgb_params.get('predictor')}")

        print("Training XGBoost...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call `.train()` first.")
        if X.ndim > 2:
            X = self.flatten(X)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if np.all(y_test == -1):
            print("Test labels unknown")
            return None
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.3f}")
        print("\nDetailed report:")
        print(classification_report(y_test, y_pred))
        return acc
