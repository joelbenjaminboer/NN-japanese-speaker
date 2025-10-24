from nomic import embed

class TimeseriesEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text-v1.5"):
        self.model = model_name

    def preprocess_timeseries(self, time_series: list[list[float]], precision: int = 2) -> str:
        """
        Preprocesses a time series into a tokenized format suitable for large language models.
        Each number is tokenized by its individual digits, a space is added between digits,
        and each time step (series) is separated by commas. The decimal point is removed, and the precision is retained.

        Args:
            time_series (list[list[float]]): A list of time steps of one channel.

        Returns:
            str: A string representing the preprocessed time series, with tokens separated by spaces and time steps separated by commas.
        """
        processed_series = []

        for time_step in time_series:
            tokenized_time_step = " ".join(
                [ " ".join([digit for digit in f"{feature:.{precision}f}".replace(".", "")]) for feature in time_step]
            )
            processed_series.append(tokenized_time_step)

        return " , ".join(processed_series)

    
    def embed(self, texts: list[str]) -> list[float]:
        """
        Embeds the preprocessed time series text using a large language model.
        Args:
            texts (str): The preprocessed time series text.
        Returns:
            list[float]: The embedding vector for the input text.
        """
        embeddings = embed.text(
            texts=texts,
            model=self.model,
            task_type='classification',
            dimensionality=64,
        )
        
        return embeddings['embeddings']

#test class
if __name__ == "__main__":
    embedder = TimeseriesEmbedder()

    # Example time series data
    time_series_data = [
        [0.12, 3.45, 6.78],
        [9.01, 2.34, 5.67],
        [8.90, 1.23, 4.56]
    ]

    # Preprocess the time series data
    preprocessed_text = embedder.preprocess_timeseries(time_series_data)
    print(preprocessed_text)

    # Embed the preprocessed text
    embeddings = embedder.embed([preprocessed_text])
    print(embeddings)