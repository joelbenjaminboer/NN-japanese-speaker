from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import numpy as np
from sklearn.preprocessing import StandardScaler

import japanese_speaker_recognition.config as cfg


# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------
def download_if_missing():
    """Download the dataset files if they are not yet in the working directory."""
    for fname in [cfg.TRAIN_FILE, cfg.TEST_FILE]:
        if not Path(fname).exists():
            print(f"Downloading {fname} ...")
            urlretrieve(cfg.DATA_URL + fname, fname)
        else:
            print(f"Found {fname} locally.")


def read_utterances(filename: str) -> List[np.ndarray]:
    """
    Read Japanese Vowels dataset, where utterances are separated by lines of 12 ones (1.0 ... 1.0).
    Returns a list of utterances, each an array of shape (T, 12).
    """
    utterances = []
    current_utt = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 12:
                continue

            try:
                floats = [float(x) for x in parts]
            except ValueError:
                continue

            # Check if separator (all ≈1.0)
            if all(abs(x - 1.0) < 1e-6 for x in floats):
                # End current utterance
                if current_utt:
                    utterances.append(np.array(current_utt, dtype=float))
                    current_utt = []
            else:
                current_utt.append(floats)

    if current_utt:
        utterances.append(np.array(current_utt, dtype=float))

    print(f"Parsed {len(utterances)} utterances from {filename}")
    return utterances


def generate_labels(num_speakers=9, utterances_per_speaker=30) -> np.ndarray:
    """
    Generate speaker labels for the training set.
    There are 9 speakers with 30 utterances each in training.
    """
    labels = []
    for speaker_id in range(num_speakers):
        labels += [speaker_id] * utterances_per_speaker
    return np.array(labels, dtype=int)


def pad_sequences(sequences: List[np.ndarray], maxlen: int) -> np.ndarray:
    """
    Pad sequences with zeros up to maxlen along the time dimension.
    Returns array of shape (n_sequences, maxlen, n_features).
    """
    n_samples = len(sequences)
    n_features = sequences[0].shape[1]
    X = np.zeros((n_samples, maxlen, n_features), dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=int)

    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        lengths[i] = T
        X[i, :T, :] = seq
    return X, lengths


def normalize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize feature-wise across time & samples using the training mean/std.
    """
    scaler = StandardScaler()
    n, t, f = X_train.shape
    X_train_2d = X_train.reshape(-1, f)
    X_test_2d = X_test.reshape(-1, f)

    scaler.fit(X_train_2d)
    X_train_norm = scaler.transform(X_train_2d).reshape(n, t, f)
    X_test_norm = scaler.transform(X_test_2d).reshape(X_test.shape[0], X_test.shape[1], f)
    return X_train_norm, X_test_norm


if __name__ == "__main__":
    download_if_missing()

    train_utts = read_utterances(cfg.TRAIN_FILE)
    print(f"Loaded {len(train_utts)} train utterances.")
    test_utts = read_utterances(cfg.TEST_FILE)
    print(f"Loaded {len(test_utts)} test utterances.")

    X_train, len_train = pad_sequences(train_utts, cfg.MAX_LEN)
    X_test, len_test = pad_sequences(test_utts, cfg.MAX_LEN)

    y_train = generate_labels()
    y_test = np.full(len(test_utts), -1)

    X_train, X_test = normalize_train_test(X_train, X_test)

    # --- Save train data ---
    np.savez_compressed(
        cfg.OUTPUT_DIR / "train_data.npz",
        X_train=X_train,
        y_train=y_train,
        len_train=len_train,
    )
    print(f"Saved training data to {cfg.OUTPUT_DIR / 'train_data.npz'}")

    # --- Save test data ---
    np.savez_compressed(
        cfg.OUTPUT_DIR / "test_data.npz",
        X_test=X_test,
        y_test=y_test,
        len_test=len_test,
    )
    print(f"Saved test data to {cfg.OUTPUT_DIR / 'test_data.npz'}")
