from pathlib import Path

from models.XGBoost import XGBoostWrapper  # use the wrapper I gave you

if __name__ == "__main__":
    DATA_DIR = Path("data/processed_data")
    aug_file = DATA_DIR / "augmented_data.npz"
    train_file = DATA_DIR / "train_data.npz"
    test_file = DATA_DIR / "test_data.npz"

    # ---------- train on augmented data ----------
    if aug_file.exists():
        print("=== XGBoost: train on AUGMENTED data ===")
        X_train, y_train, X_test, y_test = XGBoostWrapper.load_npz(aug_file, test_file)

        # keep same behavior as your RF script
        X_train = XGBoostWrapper.flatten(X_train)
        X_test = XGBoostWrapper.flatten(X_test)

        model = XGBoostWrapper(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",      # set to "gpu_hist" if you have CUDA
            random_state=42,
        )
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Test Accuracy (augmented):", (y_pred == y_test).mean())
        model.evaluate(X_test, y_test)
    else:
        print("No augmented_data.npz found — skipping augmented training.")

    # ---------- train on original data ----------
    print("\n=== XGBoost: train on ORIGINAL train data ===")
    X_train, y_train, X_test, y_test = XGBoostWrapper.load_npz(train_file, test_file)

    X_train = XGBoostWrapper.flatten(X_train)
    X_test = XGBoostWrapper.flatten(X_test)

    model = XGBoostWrapper(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",      # or "gpu_hist"
        random_state=42,
        device="cuda",
    )
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test Accuracy (original):", (y_pred == y_test).mean())
    model.evaluate(X_test, y_test)
