from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from sklearn.preprocessing import StandardScaler

# if you keep these helpers in your project:
# japanese_speaker_recognition.data_augmentation as data_aug
from config.config import Config
from japanese_speaker_recognition.data_augmentation import AugmentationPipeline
from japanese_speaker_recognition.data_embedding import EmbeddingPipeline


@dataclass
class DatasetPaths:
    data_url: str
    train_file: Path
    test_file: Path
    out_dir: Path 
    aug_file: Path | None = None

class JapaneseVowelsDataset:
    """
    End-to-end dataset handler for the Japanese Vowels dataset.

    Orchestrates:
            - download (if missing)
            - parsing utterances
            - padding to MAX_LEN
            - label generation
            - train/test normalization (fit on train only)
            - optional augmentation
            - saving NPZ artifacts
    """

    def __init__(
        self,
        cfg: Config,
        augmenter: AugmentationPipeline | None = None,
        embedding_dim: int = 64,
        embedding_model: str = "nomic-embed-text-v1.5",
        embedidng_precision: int = 2,
        key: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.augmenter = augmenter
        if key is None:
            self.key = "default"
        else:
            self.key = key

        # ---- Embedding config ----
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.embedding_precision = embedidng_precision
        

        # ---- Parse config ----
        data_url = cfg.data_url.rstrip("/") + "/"
        in_dirs = cfg.input_dirs
        out_dirs = cfg.output_dirs
        aug_cfg = cfg.augmentation 

        self.max_len: int = cfg.max_length
        self.n_features: int = cfg.n_features

        self.aug_enable: bool = aug_cfg.enabled
        self.aug_repeats: int = aug_cfg.repeats

        self.paths: DatasetPaths = DatasetPaths(
            data_url=data_url,
            train_file=in_dirs.train_file_dir,
            test_file=in_dirs.test_file_dir,
            out_dir=out_dirs.processed_file_dir,
            aug_file=aug_cfg.aug_dir,
        )
        self.paths.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_npz: Path = self.paths.out_dir / f"train_{self.key}_data.npz"
        self.test_npz: Path = self.paths.out_dir / f"test_{self.key}_data.npz"

        self.scaler: StandardScaler | None = None  # fitted on train

    # --------------------------
    # Public API
    # --------------------------
    def prepare(self) -> dict[str, Any]:
        """
        Prepare the full dataset.

        Pipeline:
          - Download & parse raw files.
          - Optionally augment training data (raw).
          - Normalize features to [0, 1] using training stats.
          - Embed time series (handles padding + fusion).
          - Save artifacts (.npz).

        Returns:
            :dict:
              - "X_train": np.ndarray
              - "y_train": np.ndarray
              - "X_test": np.ndarray
              - "y_test": np.ndarray
              - "train_npz": Path
              - "test_npz": Path
            Optional (if augmentation):
              - "X_augmented": np.ndarray
              - "y_augmented": np.ndarray
        """
        self._download_if_missing()

        # --------------------------
        # Parse raw data
        # --------------------------
        train_utts = self._read_utterances(self.paths.train_file)
        test_utts = self._read_utterances(self.paths.test_file)
        print(f"Parsed {len(train_utts)} train and {len(test_utts)} test utterances.")

        # --------------------------
        # Optional augmentation
        # --------------------------
        y_train = self._generate_train_labels()
        if self.aug_enable and self.aug_repeats > 0:
            print("Augmentation enabled.")
            if not self.augmenter:
                raise ValueError("Augmentation requested but no augmenter provided.")

            print(f"Running data augmentation ({self.aug_repeats}x repeats)...")
            # Pad sequences before augmentation
            X_train_padded, _ = self._pad_sequences(train_utts, self.max_len)
            
            X_aug, y_aug = self._augment_repeat(
                X_train_padded, y_train, repeats=self.aug_repeats
            )
            
            # Convert augmented padded arrays back to list of variable-length arrays
            # by removing padding
            train_utts_aug = []
            for i, aug_seq in enumerate(X_aug):
                # Find the actual length (before padding was added)
                # For simplicity, keep the full padded sequence or implement unpadding logic
                # Here we'll transpose back to (12, T) format
                train_utts_aug.append(aug_seq.T)  # Shape: (max_len, 12) -> (12, max_len)
            
            train_utts += train_utts_aug
            y_train = np.concatenate([y_train, y_aug])
            print(f"Augmented training set: now {len(train_utts)} utterances total.")

        # --------------------------
        # Normalize after augmentation
        # --------------------------
        print("Normalizing training and test utterances to [0, 1] range...")
        train_utts, test_utts = self._minmax_normalize_train_test(train_utts, test_utts)

        # --------------------------
        # Checking if file exists else embedding and saving
        # --------------------------
        print(self.paths.out_dir)
        print("Embedding training utterances...")
        if not os.path.exists(self.train_npz):
            train_embedder = EmbeddingPipeline(
                timeseries=train_utts,
                model_name=self.embedding_model,
                dimension=self.embedding_dim,
                precision=self.embedding_precision,
            )
            test_embedder = EmbeddingPipeline(
                timeseries=test_utts,
                model_name=self.embedding_model,
                dimension=self.embedding_dim,
                precision=self.embedding_precision,
            )
            
            X_train = train_embedder.get_fused
            X_test = test_embedder.get_fused
            
            y_train = self._generate_train_labels()
            y_test = self._generate_test_labels()
            
            np.savez_compressed(self.train_npz, X_train=X_train, y_train=y_train)
            np.savez_compressed(self.test_npz, X_test=X_test, y_test=y_test)
            
            out = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_npz": str(self.train_npz),
            "test_npz": str(self.test_npz),
            }

            if self.aug_enable and self.aug_repeats > 0:
                out.update(
                    {
                        "X_augmented": X_train[-len(y_aug) :],
                        "y_augmented": y_aug,
                    }
                )

            return out

        # if artifacts exist, load and return them
        else:
            print(f"Loading existing embeddings from: {self.train_npz}")
            print(f"Loading existing embeddings from: {self.test_npz}")
            
            X_train = np.load(self.train_npz)["X_train"]
            X_test = np.load(self.test_npz)["X_test"]
            y_train = np.load(self.train_npz)["y_train"]
            y_test = np.load(self.test_npz)["y_test"]
            
            out = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_npz": str(self.train_npz),
            "test_npz": str(self.test_npz),
            }

            if self.aug_enable and self.aug_repeats > 0:
                out.update(
                    {
                        "X_augmented": X_train[-len(y_aug) :],
                        "y_augmented": y_aug,
                    }
                )

            return out

    # --------------------------
    # Steps (private)
    # --------------------------
    def _download_if_missing(self) -> None:
        for fname in [self.paths.train_file.name, self.paths.test_file.name]:
            dest = self.paths.train_file.parent / fname
            if not dest.exists():
                url = self.paths.data_url + fname
                print(f"Downloading {url} -> {dest}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(url, dest)
            else:
                print(f"Found {dest}")

    def _read_utterances(self, filename: Path) -> list[np.ndarray]:
        """
        Reads the Japanese Vowels dataset.
        Each utterance is transposed to shape (12, T), with variable T (no padding).
        Returns: list of np.ndarray, each of variable length in time.
        """
        utterances: list[np.ndarray] = []
        current_utt: list[list[float]] = []

        with open(filename, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != self.n_features:
                    continue
                try:
                    floats = [float(x) for x in parts]
                except ValueError:
                    continue

                if all(abs(x - 1.0) < 1e-6 for x in floats):
                    if current_utt:
                        utt = np.array(current_utt, dtype=float).T
                        utterances.append(utt)
                        current_utt = []
                else:
                    current_utt.append(floats)

        if current_utt:
            utterances.append(np.array(current_utt, dtype=float).T)

        return utterances

    def _pad_sequences(
        self, sequences: list[np.ndarray], maxlen: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples = len(sequences)
        n_features = sequences[0].shape[0]  # First dimension is features
        X = np.zeros((n_samples, maxlen, n_features), dtype=np.float32)
        lengths = np.zeros(n_samples, dtype=int)

        for i, seq in enumerate(sequences):
            T = seq.shape[1]  # Second dimension is time
            lengths[i] = T
            # Transpose from (n_features, T) to (T, n_features) before storing
            X[i, :T, :] = seq.T
        return X, lengths

    def _minmax_normalize_train_test(
        self, train_utts: list[np.ndarray], test_utts: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Min-max normalize each feature dimension of all utterances to [0, 1],
        using min and max computed across *all* training utterances.
        Args:
            train_utts (list[np.ndarray]): list of arrays (12, T_i) for training set
            test_utts  (list[np.ndarray]): list of arrays (12, T_j) for test set
        Returns:
            tuple: (normalized_train_utts, normalized_test_utts)
        """

        all_train_concat = np.concatenate(train_utts, axis=1)
        feat_min = all_train_concat.min(axis=1, keepdims=True)
        feat_max = all_train_concat.max(axis=1, keepdims=True)
        denom = np.where(feat_max - feat_min == 0, 1e-8, feat_max - feat_min)

        def normalize_list(utts: list[np.ndarray]) -> list[np.ndarray]:
            normalized = []
            for utt in utts:
                utt_norm = (utt - feat_min) / denom
                normalized.append(utt_norm)
            return normalized

        norm_train = normalize_list(train_utts)
        norm_test = normalize_list(test_utts)

        self.scaler = {"min": feat_min, "max": feat_max}

        return norm_train, norm_test

    def _normalize_train_test(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        n, t, f = X_train.shape
        Xtr2 = X_train.reshape(-1, f)
        Xte2 = X_test.reshape(-1, f)
        scaler.fit(Xtr2)
        X_train_norm = scaler.transform(Xtr2).reshape(n, t, f)
        X_test_norm = scaler.transform(Xte2).reshape(X_test.shape[0], X_test.shape[1], f)
        self.scaler = scaler
        return X_train_norm, X_test_norm

    # Label helpers (dataset-specific)
    @staticmethod
    def _generate_train_labels(
        num_speakers: int = 9, utterances_per_speaker: int = 30
    ) -> np.ndarray:
        labels: list[int] = []
        for speaker_id in range(num_speakers):
            labels += [speaker_id] * utterances_per_speaker
        return np.array(labels, dtype=int)

    @staticmethod
    def _generate_test_labels() -> np.ndarray:
        block_sizes = [31, 35, 88, 44, 29, 24, 40, 50, 29]
        labels: list[int] = []
        for speaker_id, n in enumerate(block_sizes):
            labels += [speaker_id] * n
        return np.array(labels, dtype=int)

    # Augmentation
    def _augment_repeat(self, X_train: np.ndarray, y_train: np.ndarray, repeats: int):
        if not self.augmenter or repeats <= 0:
            return np.empty((0, *X_train.shape[1:])), np.empty((0,), dtype=y_train.dtype)
        X_aug, idx_map = self.augmenter.run(X_train, repeats=repeats)
        y_aug = np.tile(y_train, repeats)
        return X_aug, y_aug
