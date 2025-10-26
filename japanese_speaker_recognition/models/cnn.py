import torch
import torch.nn as nn


class SpeakerCNN(nn.Module):
    """
    CNN for extracting local patterns from speaker embeddings.

    Input: [B, 12, 64] - 12 features, each with 64-dim embedding
    Output: [B, num_classes] - speaker classification logits

    Architecture:
    - 1D convolutions over the feature dimension (64) to capture temporal patterns
    - Cross-channel convolutions to capture co-movements across the 12 features
    - Small MLP classifier on top of extracted features
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # Input: [B, 12, 64]
        # Treat as [B, channels=12, sequence_length=64]

        # 1D convolutions over temporal dimension (64)
        # Look for local patterns, spikes, ramps in embeddings
        self.temporal_conv = nn.Sequential(
            # Conv1: capture short-range patterns (kernel=3)
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Conv2: capture medium-range patterns (kernel=5)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [B, 64, 32]
            nn.Dropout(dropout),
            # Conv3: capture longer patterns (kernel=5)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [B, 128, 16]
            nn.Dropout(dropout),
        )

        # Global pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, 128, 1]

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 128]
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 12, 64]

        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        # Extract temporal patterns
        x = self.temporal_conv(x)  # [B, 128, 16]

        # Global pooling
        x = self.global_pool(x)  # [B, 128, 1]

        # Classify
        logits = self.classifier(x)  # [B, num_classes]

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features without classification."""
        x = self.temporal_conv(x)
        x = self.global_pool(x)
        return x.squeeze(-1)  # [B, 128]
