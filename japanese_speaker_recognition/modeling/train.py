from pathlib import Path
import numpy as np
from models.random_forest import RandomForestWrapper

if __name__ == "__main__":
    DATA_DIR = Path("data/processed_data")
    train_file = DATA_DIR / "train_data.npz"
    test_file = DATA_DIR / "test_data.npz"

    rf = RandomForestWrapper(n_estimators=300, random_state=42)
    X_train, y_train, X_test, y_test = rf.load_npz(train_file, test_file)

    X_train_flat = rf.flatten(X_train)
    X_test_flat = rf.flatten(X_test)

    rf.train(X_train_flat, y_train)
    rf.evaluate(X_test_flat, y_test)