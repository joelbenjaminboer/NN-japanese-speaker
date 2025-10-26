import torch
import torch.nn as nn


class HAIKU(nn.Module):
    """
    HAIKU — Hybrid Add-embedding + Inference with Kernels (CNN) Unit

    Input: [B, 12, embedding_dim] - 12 features, each with embedding_dim
    Output: [B, num_classes] - speaker classification logits

    Architecture:
    - 1 convolutional layer over temporal dimension (CNN feature extractor)
    - 2 linear layers for classification (MLP classifier head)

    Args:
            num_classes: Number of output classes
            dropout: Dropout probability
            embedding_dim: Dimension of input embeddings
            kernel_size: Kernel size for conv layer
            conv_channels: Number of output channels for conv layer
            hidden_dim: Hidden dimension for first linear layer
            input_channels: Number of input channels (features)
    """

     def __init__(
        self,
        num_classes: int = 9,
        dropout: float = 0.3,
        embedding_dim: int = 64,
        kernel_size: int = 3,
        conv_channels: int = 128,
        hidden_dim: int = 64,
        input_channels: int = 12
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Single convolutional layer
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=conv_channels, 
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout))

        # Global pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, conv_channels, 1]

        # MLP: 2 linear layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, conv_channels, 1] -> [B, conv_channels]
            nn.Linear(conv_channels, hidden_dim),  # First linear layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # Second linear layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 12, 64]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        # Extract temporal patterns
        x = self.conv(x)  # [B, 128, 16]

        # Global pooling
        x = self.global_pool(x)  # [B, 128, 1]

        # Classify
        logits = self.classifier(x)  # [B, num_classes]

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features without classification."""
        x = self.conv(x)
        x = self.global_pool(x)
        return x.squeeze(-1)  # [B, 128]

    @classmethod
    def from_config(cls, config: dict):
        """
        Create SpeakerCNN from configuration dictionary.
        Args:
            config: Dictionary with CNN configuration
        Returns:
            SpeakerCNN instance
        """
        return cls(
            dropout=config.get("DROPOUT", 0.3),
            embedding_dim=config.get("EMBEDDING_DIM", 64),
            kernel_size=config.get("KERNEL_SIZE", 5),
            conv_channels=config.get("CONV_CHANNELS", 128),
            hidden_dim=config.get("HIDDEN_DIM", 64)
        )
