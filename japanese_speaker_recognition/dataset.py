from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from sklearn.preprocessing import StandardScaler

# if you keep these helpers in your project:
# japanese_speaker_recognition.data_augmentation as data_aug
from japanese_speaker_recognition.data_augmentation import AugmentationPipeline


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

	def __init__(self, cfg: dict[str, Any], augmenter: AugmentationPipeline | None = None) -> None:
		self.cfg = cfg
		self.augmenter = augmenter

		# ---- Parse config ----
		data_url = cfg.get("DATA_URL", "").rstrip("/") + "/"
		in_dirs = cfg.get("INPUT_DIRS", {})
		out_dirs = cfg.get("OUTPUT_DIRS", {})
		aug_cfg = cfg.get("AUGMENTATION", {}) or {}

		self.max_len: int = int(cfg.get("MAX_LEN", 29))
		self.n_features: int = int(cfg.get("N_FEATURES", 12))

		self.pipeline_flags = {
			"train": bool(cfg.get("PIPELINE", {}).get("TRAIN", True)),
			"test": bool(cfg.get("PIPELINE", {}).get("TEST", False)),
			"augment": bool(cfg.get("PIPELINE", {}).get("AUGMENT", False)),
		}

		self.aug_enable: bool = bool(aug_cfg.get("AUGMENT", False))
		self.aug_repeats: int = int(
			aug_cfg.get("REPEATS", 0)
		)  # optional; defaults to 0 if not in YAML

		paths = DatasetPaths(
			data_url=data_url,
			train_file=Path(in_dirs.get("TRAIN_FILE", "data/ae.train")),
			test_file=Path(in_dirs.get("TEST_FILE", "data/ae.test")),
			out_dir=Path(out_dirs.get("PROCESSED", "data/processed_data")),
			aug_file=Path(aug_cfg.get("AUG_FILE")) if aug_cfg.get("AUG_FILE") else None,
		)
		self.paths = paths
		self.paths.out_dir.mkdir(parents=True, exist_ok=True)

		self.scaler: StandardScaler | None = None  # fitted on train

	# --------------------------
	# Public API
	# --------------------------
	def prepare(self) -> dict[str, Any]:
		"""
		Runs the configured parts of the pipeline and returns a dict of arrays.
		Also writes NPZ files under OUTPUT_DIRS.PROCESSED.
		"""
		self._download_if_missing()

		# Parse raw
		train_utts = self._read_utterances(self.paths.train_file)
		test_utts = self._read_utterances(self.paths.test_file)

		# Pad
		X_train, len_train = self._pad_sequences(train_utts, self.max_len)
		X_test, len_test = self._pad_sequences(test_utts, self.max_len)

		# Labels
		y_train = self._generate_train_labels()
		y_test = self._generate_test_labels()

		# Normalize (fit on train only)
		X_train, X_test = self._normalize_train_test(X_train, X_test)

		# Save core artifacts
		train_npz = self.paths.out_dir / "train_data.npz"
		test_npz = self.paths.out_dir / "test_data.npz"

		np.savez_compressed(train_npz, X_train=X_train, y_train=y_train, len_train=len_train)
		np.savez_compressed(test_npz, X_test=X_test, y_test=y_test, len_test=len_test)

		out = {
			"X_train": X_train,
			"y_train": y_train,
			"len_train": len_train,
			"X_test": X_test,
			"y_test": y_test,
			"len_test": len_test,
			"train_npz": str(train_npz),
			"test_npz": str(test_npz),
		}

		# Optional augmentation
		if self.pipeline_flags["augment"] and self.aug_enable and self.aug_repeats > 0:
			X_aug, y_aug = self._augment_repeat(X_train, y_train, repeats=self.aug_repeats)
			# prefer AUG_FILE from config; otherwise drop inside processed dir
			aug_path = self.paths.aug_file or (self.paths.out_dir / "augmented_data.npz")
			np.savez_compressed(aug_path, X_augmented=X_aug, y_augmented=y_aug)
			out.update({"X_augmented": X_aug, "y_augmented": y_aug, "augmented_npz": str(aug_path)})

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
		Read Japanese Vowels dataset, where utterances are separated by lines of 12 ones \
			(1.0 ... 1.0).
		Returns a list of utterances, each an array of shape (T, 12).
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

				# separator if all ~1.0
				if all(abs(x - 1.0) < 1e-6 for x in floats):
					if current_utt:
						utterances.append(np.array(current_utt, dtype=float))
						current_utt = []
				else:
					current_utt.append(floats)

		if current_utt:
			utterances.append(np.array(current_utt, dtype=float))

		print(f"Parsed {len(utterances)} utterances from {filename}")
		return utterances

	def _pad_sequences(
		self, sequences: list[np.ndarray], maxlen: int
	) -> tuple[np.ndarray, np.ndarray]:
		n_samples = len(sequences)
		n_features = sequences[0].shape[1]
		X = np.zeros((n_samples, maxlen, n_features), dtype=np.float32)
		lengths = np.zeros(n_samples, dtype=int)

		for i, seq in enumerate(sequences):
			T = seq.shape[0]
			lengths[i] = T
			X[i, :T, :] = seq
		return X, lengths

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
	def _generate_train_labels(
		self, num_speakers: int = 9, utterances_per_speaker: int = 30
	) -> np.ndarray:
		labels: list[int] = []
		for speaker_id in range(num_speakers):
			labels += [speaker_id] * utterances_per_speaker
		return np.array(labels, dtype=int)

	def _generate_test_labels(self) -> np.ndarray:
		block_sizes = [31, 35, 88, 44, 29, 24, 40, 50, 29]  # per speaker 1–9
		labels: list[int] = []
		for speaker_id, n in enumerate(block_sizes):
			labels += [speaker_id] * n
		return np.array(labels, dtype=int)

	# Augmentation
	def _augment_repeat(self, X_train: np.ndarray, y_train: np.ndarray, repeats: int):
		if not self.augmenter or repeats <= 0:
			return np.empty((0, *X_train.shape[1:])), np.empty((0,), dtype=y_train.dtype)
		X_aug, idx_map = self.augmenter.run(X_train, repeats=repeats)
		y_aug = np.tile(y_train, repeats)  # keeps label-per-sample semantics
		return X_aug, y_aug
