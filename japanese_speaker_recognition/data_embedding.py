from nomic import embed
import numpy as np

class add_time_series_to_embedding:
    def __init__(self, model_name: str = "nomic-embed-text-v1.5", timeseries: list[list[float]] = None):
        self.model = model_name
        self.original_timeseries = timeseries
        self.processed_timeseries = self._preprocess_timeseries(timeseries)
        self.embeddings = self._embed(self.processed_timeseries)
        self.fused = self._fuse_embeddings_timeseries()

    @staticmethod
    def _preprocess_timeseries(self, time_series: list[list[float]], precision: int = 2) -> list[str]:
        """
        Convert a multi-channel time series (list of channels) into digit-tokenized format
        described in Gruver et al. (2024). Each channel is tokenized separately.

        Input shape:  [n_channels][n_timesteps]
        Output shape: [n_channels]  (each entry is a comma-separated token string)

        Example:
            Input:
                [
                    [0.645, 6.45, 64.5, 645.0],  # channel 1
                    [1.23, 12.3, 123.0, 1230.0]  # channel 2
                ]

            Output:
                [
                    "6 4 , 6 4 5 , 6 4 5 0 , 6 4 5 0 0",
                    "1 2 3 , 1 2 3 0 , 1 2 3 0 0 , 1 2 3 0 0 0"
                ]
        """
        processed_channels = []
        for channel in time_series:
            channel_tokens = []
            for value in channel:
                scaled = int(round(value * (10 ** precision)))
                digits = " ".join(list(str(scaled)))
                channel_tokens.append(digits)
            processed_channels.append(" , ".join(channel_tokens))
        return processed_channels

    @staticmethod
    def _embed(self, texts: list[str], dimension: int = 64) -> list[list[float]]:
        """
        Embeds the preprocessed time series text using a large language model.
        args:
            texts: List of preprocessed time series strings.
        returns:
            List of embeddings corresponding to each input text.
            list[list[float_channel_1], list[float_channel 2], ...]
        """
        if texts is None:
            raise ValueError("Input texts for embedding cannot be None.")
        
        embeddings = embed.text(
            texts=texts,
            model=self.model,
            task_type='classification',
            dimensionality=dimension,
        )
        
        return np.array(embeddings['embeddings'], dtype=float)
    
    @staticmethod
    def _fuse_embeddings_timeseries(self) -> np.ndarray:
        """
        Fuses raw time series (n_channels x n_timesteps)
        with embeddings (n_channels x embedding_dim)
        using element-wise addition and zero padding
        for dimensional consistency.

        Returns:
            fused (np.ndarray): fused representation of shape (n_channels, max(n_timesteps, embedding_dim))
        """
        if self.original_timeseries is None or self.embeddings is None:
            raise ValueError("Original timeseries or embeddings not initialized.")

        ts = np.array(self.original_timeseries, dtype=float)
        emb = np.array(self.embeddings, dtype=float)

        n_timesteps = ts.shape[1]
        embedding_dim = emb.shape[1]

        fused_length = max(n_timesteps, embedding_dim)

        def pad_to(arr, target_len):
            pad_width = target_len - arr.shape[1]
            if pad_width > 0:
                return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
            else:
                return arr[:, :target_len]
       
        ts_padded = pad_to(ts, fused_length)
        emb_padded = pad_to(emb, fused_length)

        fused = ts_padded + emb_padded
        return fused

    @property
    def get_embeddings(self) -> np.ndarray:
        return self.embeddings
    
    @property
    def get_fused(self) -> np.ndarray:
        return self.fused
        
        

