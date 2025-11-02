import time

import numpy as np
import torch
from nomic import embed
from sentence_transformers import SentenceTransformer


class BaseEmbeddingPipeline:
    """Base class with shared functionality for embedding pipelines."""
    
    def __init__(self, timeseries: list[np.ndarray], dimension: int, precision: int):
        """Initialize base embedding pipeline."""
        if not isinstance(timeseries, list):
            raise ValueError("Expected list of np.ndarray, each shaped (12, T_i)")
        
        self.dimension = dimension
        self.precision = precision
        self.original_timeseries = timeseries
    
    def _preprocess_timeseries(self, utterance: np.ndarray, task_prefix: str = "") -> list[str]:
        """
        Converts a single utterance (12 x T) into digit-tokenized text per channel.
        
        Args:
            utterance: Array of shape (12, T)
            task_prefix: Optional prefix for the text (e.g., "classification: ")
        """
        processed_channels = []
        for channel in utterance:
            channel_tokens = []
            for value in channel:
                scaled = int(round(value * (10**self.precision)))
                digits = " ".join(list(str(scaled)))
                channel_tokens.append(digits)
            
            channel_text = " , ".join(channel_tokens)
            if task_prefix:
                channel_text = f"{task_prefix}: {channel_text}"
            
            processed_channels.append(channel_text)
        return processed_channels
    
    @staticmethod
    def _fuse_embeddings_timeseries(timeseries: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Fuses raw time series (12 x T_i) with embeddings (12 x embedding_dim)
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
    
    def save_fused(self, filepath: str) -> list[np.ndarray]:
        """Saves fused representations to a .npz file."""
        np.savez_compressed(filepath, *self.fused)
        return self.fused
    
    @property
    def get_embeddings(self) -> list[np.ndarray]:
        """List of embeddings (each np.ndarray of shape (12, embedding_dim))"""
        return self.embeddings

    @property
    def get_fused(self) -> list[np.ndarray]:
        """List of fused representations (each np.ndarray of shape (12, max(T, embedding_dim)))"""
        return self.fused


class LocalEmbeddingPipeline(BaseEmbeddingPipeline):
    """
    Local embedding pipeline using sentence-transformers.
    Designed for offline HPC environments without internet access.
    Uses nomic-embed-text-v1.5 with GPU acceleration.
    """

    def __init__(
        self,
        timeseries: list[np.ndarray],
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        dimension: int = 64,
        precision: int = 2,
        device: str = "auto",
        batch_size: int = 32,
        task_prefix: str = "classification",
    ) -> None:
        """
        Embedding pipeline for a list of utterances (each np.ndarray of shape (12, T_i)).
        Converts each utterance to a digit-tokenized representation,
        embeds each channel, and fuses embeddings with original data.

        Args:
            timeseries: List of utterances, each shaped (12, T_i)
            model_name: HuggingFace model identifier
            dimension: Target embedding dimension (Matryoshka)
            precision: Decimal precision for digit tokenization
            device: Device to run on ('auto', 'cuda', 'cpu')
            batch_size: Batch size for embedding (32 is good for H100 10GB, 
                       increase to 64-128 for larger GPUs)
            task_prefix: Task instruction prefix for nomic embeddings
                        ('classification', 'clustering', 'search_document', 'search_query')
        """
        super().__init__(timeseries, dimension, precision)
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.task_prefix = task_prefix

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=self.device,
        )
        print(f"Model loaded successfully on {self.device}")

        # Run the embedding pipeline
        self.processed_timeseries = [
            self._preprocess_timeseries(utt, task_prefix=self.task_prefix) 
            for utt in timeseries
        ]
        self.embeddings = self._embed(self.processed_timeseries)
        self.fused = [
            self._fuse_embeddings_timeseries(utt, emb)
            for utt, emb in zip(timeseries, self.embeddings, strict=True)
        ]

    def _embed(self, processed_utterances: list[list[str]]) -> list[np.ndarray]:
        """
        Efficient batched embedding using sentence-transformers locally.
        Applies Matryoshka dimensionality reduction as per nomic v1.5 spec.
        """
        # Flatten list of utterance-channel strings
        all_texts = [text for utt in processed_utterances for text in utt]
        n_utts = len(processed_utterances)
        n_channels = len(processed_utterances[0])

        print(f"Embedding {len(all_texts)} texts ({n_utts} utterances × {n_channels} channels)")

        all_embs = []
        for i in range(0, len(all_texts), self.batch_size):
            chunk = all_texts[i : i + self.batch_size]
            print(
                f"Embedding batch {i // self.batch_size + 1}/{-(-len(all_texts) // self.batch_size)} "
                f"({len(chunk)} texts)"
            )

            # Encode with sentence-transformers
            embeddings = self.model.encode(
                chunk,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )

            # Apply Matryoshka dimensionality reduction (as per nomic v1.5 docs)
            # 1. Layer normalization
            embeddings = torch.nn.functional.layer_norm(
                embeddings, normalized_shape=(embeddings.shape[1],)
            )
            # 2. Truncate to desired dimension
            embeddings = embeddings[:, : self.dimension]
            # 3. L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Convert to numpy
            emb_np = embeddings.cpu().numpy()
            all_embs.append(emb_np)

        # Concatenate all batches
        all_embs = np.concatenate(all_embs, axis=0)
        print(f"Total embeddings generated: {all_embs.shape}")

        # Reshape back to list[np.ndarray] per utterance (12 × embedding_dim)
        reshaped = [all_embs[i * n_channels : (i + 1) * n_channels] for i in range(n_utts)]
        return reshaped


class EmbeddingPipeline(BaseEmbeddingPipeline):
    """
    API-based embedding pipeline using Nomic's hosted service.
    Requires internet connection and Nomic API key.
    """
    
    def __init__(
        self,
        timeseries: list[np.ndarray],
        model_name: str = "nomic-embed-text-v1.5",
        dimension: int = 64,
        precision: int = 2,
    ) -> None:
        """
        Embedding pipeline for a list of utterances (each np.ndarray of shape (12, T_i)).
        Converts each utterance to a digit-tokenized representation,
        embeds each channel, and fuses embeddings with original data.
        """
        super().__init__(timeseries, dimension, precision)
        
        self.model = model_name

        # Run the embedding pipeline
        self.processed_timeseries = [
            self._preprocess_timeseries(utt) 
            for utt in timeseries
        ]
        self.embeddings = self._embed(self.processed_timeseries)
        self.fused = [
            self._fuse_embeddings_timeseries(utt, emb)
            for utt, emb in zip(timeseries, self.embeddings, strict=True)
        ]
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
        batch_size = 100  # <= recommended by Nomic
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
