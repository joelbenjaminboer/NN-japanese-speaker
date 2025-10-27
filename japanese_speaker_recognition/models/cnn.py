from typing import Self

from numpy import ndarray

import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.pooling import AdaptiveAvgPool1d
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from utils.utils import heading


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
        input_channels: int = 12,
        device: str = "cpu",
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.device = self._get_device(device)

        # Single convolutional layer
        padding = kernel_size // 2
        self.convolution: Sequential = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Global pooling to get fixed-size representation
        self.global_pool: AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)  # [B, conv_channels, 1]

        # MLP: 2 linear layers for classification
        self.perceptron: Sequential = nn.Sequential(
            nn.Flatten(),  # [B, conv_channels, 1] -> [B, conv_channels]
            nn.Linear(conv_channels, hidden_dim),  # First linear layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),  # Second linear layer
        )


    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 12, 64]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        # Extract temporal patterns
        x = self.convolution(x)  # [B, 128, 16]

        # Global pooling
        x = self.global_pool(x)  # [B, 128, 1]

        # Classify
        logits = self.perceptron(x)  # [B, num_classes]

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features without classification."""
        x = self.convolution(x)
        x = self.global_pool(x)
        return x.squeeze(-1)  # [B, 128]

    @classmethod
    def _from_config(cls, config: dict[str, int | float]) -> Self:
        """
        Create SpeakerCNN from configuration dictionary.
        Args:
            config: Dictionary with CNN configuration
        Returns:
            SpeakerCNN instance
        """
        return cls(
            dropout=float(config.get("DROPOUT", 0.3)),
            embedding_dim=int(config.get("EMBEDDING_DIM", 64)),
            kernel_size=int(config.get("KERNEL_SIZE", 5)),
            conv_channels=int(config.get("CONV_CHANNELS", 128)),
            hidden_dim=int(config.get("HIDDEN_DIM", 64)),
            device=str(config.get("DEVICE", "cpu")),
        )
    
    @staticmethod
    def _get_device(device: str = "auto") -> str:
        """
        Get the device to use for training.

        Args:
            device: Device specification from config ("cuda", "cpu", or "auto")

        Returns:
            Device string ("cuda" or "cpu")
        """

        heading("Device Configuration")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
        else:
            device = "cpu"

        if device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Using CPU")

        return device

    @classmethod
    def create_model(cls, model_cfg: dict) -> Self:
        """Create CNN model from configuration."""
        model = cls._from_config(model_cfg)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model created:")
        print(
            f"  - Input: [Batch, {model_cfg.get('INPUT_CHANNELS', 12)}, \
            {model_cfg.get('EMBEDDING_DIM', 64)}]"
        )
        print(f"  - Conv channels: {model_cfg.get('CONV_CHANNELS', 128)}")
        print(f"  - Kernel size: {model_cfg.get('KERNEL_SIZE', 3)}")
        print(f"  - Hidden dim: {model_cfg.get('HIDDEN_DIM', 64)}")
        print(f"  - Output: [Batch, {model_cfg.get('NUM_CLASSES', 9)}]")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        return model

    def train_model(
        self,
        x_train: ndarray,
        y_train: ndarray,
        x_val: ndarray,
        y_val: ndarray,
        learning_rate: float = 0.007,
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> dict:
        """Train the model and return training history."""

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.RAdam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.to(self.device)

        heading(f"Training for {num_epochs} epochs (lr={learning_rate}, device={self.device})")

        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self._train_step(train_loader, optimizer, criterion)

            # Validation
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            else:
                val_loss, val_acc = 0.0, 0.0
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # Store history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        return history

    def _train_step(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Perform one training epoch."""
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def evaluate(
        self, dataloader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """Evaluate model on validation/test set."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
