import time
import os
import numpy as np
from nomic import embed

class EmbeddingPipeline:
    
    def __init__(
        self,
        timeseries: list[np.ndarray],
        model_name: str = "nomic-embed-text-v1.5",
        dimension: int = 64,
        precision: int = 2,
        test_dir: str = "data/processed_data/",
        train_dir: str = "data/processed_data/",
        key: str = None,
    ) -> None:
        """
        Embedding pipeline for a list of utterances (each np.ndarray of shape (12, T_i)).
        Converts each utterance to a digit-tokenized representation,
        embeds each channel, and fuses embeddings with original data.
        """
        if not isinstance(timeseries, list):
            raise ValueError("Expected list of np.ndarray, each shaped (12, T_i)")

        self.model = model_name
        self.dimension = dimension
        self.precision = precision
        self.original_timeseries = timeseries
        
        if not os.path.exists(os.path.join(test_dir, f"test_{key}_fused.npy")):
            # Run the embedding pipeline
            self.processed_timeseries = [self._preprocess_timeseries(utt) for utt in timeseries]
            self.embeddings = self._embed(self.processed_timeseries)
            self.fused = [
                self._fuse_embeddings_timeseries(utt, emb)
                for utt, emb in zip(timeseries, self.embeddings, strict=True)
            ]
            # Save fused representations
            np.save(os.path.join(test_dir, f"test_{key}_fused.npy"), self.fused)
        else:
            print("Fused representations already exist.")
            self.fused = np.load(os.path.join(test_dir, f"test_{key}_fused.npy"), allow_pickle=True)
        
        if not os.path.exists(os.path.join(train_dir, f"test_{key}_fused.npy")):
            # Run the embedding pipeline
            self.processed_timeseries = [self._preprocess_timeseries(utt) for utt in timeseries]
            self.embeddings = self._embed(self.processed_timeseries)
            self.fused = [
                self._fuse_embeddings_timeseries(utt, emb)
                for utt, emb in zip(timeseries, self.embeddings, strict=True)
            ]
            # Save fused representations
            np.save(os.path.join(train_dir, f"train_{key}_fused.npy"), self.fused)
        else:
            print("Fused representations already exist.")
            self.fused = np.load(os.path.join(train_dir, f"train_{key}_fused.npy"), allow_pickle=True)


    # -----------------------------
    # Step 1: preprocessing
    # -----------------------------
    def _preprocess_timeseries(self, utterance: np.ndarray) -> list[str]:
        """
        Converts a single utterance (12 x T) into digit-tokenized text per channel.
        """
        processed_channels = []
        for channel in utterance:
            channel_tokens = []
            for value in channel:
                scaled = int(round(value * (10**self.precision)))
                digits = " ".join(list(str(scaled)))
                channel_tokens.append(digits)
            processed_channels.append(" , ".join(channel_tokens))
        return processed_channels

    # -----------------------------
    # Step 2: embedding
    # -----------------------------
    def _embed(self, processed_utterances: list[list[str]]) -> list[np.ndarray]:
        """
        Efficient batched embedding with automatic rate-limit handling.
        Splits requests into manageable chunks and retries on 429 responses.
        """
        # Flatten list of utterance-channel strings
        all_texts = [text for utt in processed_utterances for text in utt]
        n_utts = len(processed_utterances)
        n_channels = len(processed_utterances[0])

        all_embs = []
        batch_size = 100
        for i in range(0, len(all_texts), batch_size):
            chunk = all_texts[i : i + batch_size]

            # Retry loop with exponential backoff
            for attempt in range(5):
                try:
                    print(
                        f"Embedding chunk {i // batch_size + 1}/{-(-len(all_texts) // batch_size)} \
                            ({len(chunk)})"
                    )
                    response = embed.text(
                        texts=chunk,
                        model=self.model,
                        dimensionality=self.dimension,
                    )
                    emb = np.array(response["embeddings"], dtype=float)
                    all_embs.append(emb)
                    break  # success, exit retry loop
                except Exception as e:
                    if "429" in str(e):
                        wait = 5 * (attempt + 1)
                        print(f"Rate-limited (429). Waiting {wait}s before retry...")
                        time.sleep(wait)
                        continue
                    else:
                        raise
            else:
                raise RuntimeError("Embedding failed repeatedly due to API throttling.")

        # Concatenate all batches
        all_embs = np.concatenate(all_embs, axis=0)
        print(f"Total embeddings received: {all_embs.shape}")

        # Reshape back to list[np.ndarray] per utterance (12 × embedding_dim)
        reshaped = [all_embs[i * n_channels : (i + 1) * n_channels] for i in range(n_utts)]
        return reshaped

    # -----------------------------
    # Step 3: fusion
    # -----------------------------
    @staticmethod
    def _fuse_embeddings_timeseries(timeseries: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Fuses raw time series (12 x T_i)
        with embeddings (12 x embedding_dim)
        using element-wise addition and zero padding.
        """
        ts = np.array(timeseries, dtype=float)
        emb = np.array(embeddings, dtype=float)

        n_timesteps = ts.shape[1]
        embedding_dim = emb.shape[1]
        fused_length = max(n_timesteps, embedding_dim)

        def pad_to(arr, target_len):
            pad_width = int(target_len - arr.shape[1])
            if pad_width > 0:
                return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
            else:
                return arr[:, :target_len]

        ts_padded = pad_to(ts, fused_length)
        emb_padded = pad_to(emb, fused_length)
        fused = ts_padded + emb_padded
        return fused

    # -----------------------------
    # Accessors
    # -----------------------------
    @property
    def get_embeddings(self) -> list[np.ndarray]:
        """List of embeddings (each np.ndarray of shape (12, embedding_dim))"""
        return self.embeddings

    @property
    def get_fused(self) -> list[np.ndarray]:
        """List of fused representations (each np.ndarray of shape (12, max(T, embedding_dim)))"""
        return self.fused
