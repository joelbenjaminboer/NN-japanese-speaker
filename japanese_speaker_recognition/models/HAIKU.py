from typing import Self

import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# from sklearn.model_selection._split import KFold
from torch.nn.modules.container import Sequential
from torch.nn.modules.pooling import AdaptiveAvgPool1d
from torch.utils.data import DataLoader, Subset, TensorDataset

# from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing_extensions import override

from config.config import Model
from utils.utils import heading

# BatchData: TypeAlias = tuple[torch.Tensor, torch.Tensor]
# TrainDataset: TypeAlias = TensorDataset | Subset[TensorDataset]

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
        self.device: str = self._get_device(device)

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
    def _from_config(cls, model_cfg: Model) -> Self:
        """
        Create SpeakerCNN from configuration dictionary.
        Args:
            config: Dictionary with CNN configuration
        Returns:
            SpeakerCNN instance
        """
        return cls(
            dropout=model_cfg.dropout,
            embedding_dim=model_cfg.embedding_dim,
            kernel_size=model_cfg.kernel_size,
            conv_channels=model_cfg.conv_channels,
            hidden_dim=model_cfg.hidden_dim,
            device=model_cfg.device,
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

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    @classmethod
    def create_model(cls, model_cfg: Model) -> Self:
        """Create HAIKU model from configuration."""
        model = cls._from_config(model_cfg)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model created:")
        print(
            f"  - Input: [Batch, {model_cfg.input_channels}, \
            {model_cfg.embedding_dim}]"
        )
        print(f"  - Conv channels: {model_cfg.conv_channels}")
        print(f"  - Kernel size: {model_cfg.kernel_size}")
        print(f"  - Hidden dim: {model_cfg.hidden_dim}")
        print(f"  - Output: [Batch, {model_cfg.num_classes}]")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        return model

    def train_model(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 0.007,
        num_epochs: int = 10,
        batch_size: int = 32,
        k_folds: int = 5,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42
    ) -> tuple[dict[str, list[float]], dict[str, float]]:
        """Train the model using K-Fold cross-validation and return averaged training history."""

        torch.manual_seed(seed)
        self.to(self.device)

        heading(f"Training for {num_epochs} epochs with {k_folds}-Fold cross-validation \
            (lr={learning_rate}, device={self.device})")

        full_dataset = TensorDataset(x_train, y_train)
        kf: KFold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        # Track per-fold averaged metrics
        fold_averaged_history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }
        
        # Track all epoch metrics across all folds
        global_history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

        # Loop through folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
            print(f"\n{'='*20} Fold {fold + 1}/{k_folds} {'='*20}")
            
            # Split dataset
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
            )
            
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # Reinitialize model weights before each fold
            self = self.apply(self._init_weights)

            optimizer = torch.optim.RAdam(self.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Track metrics for THIS fold only
            fold_history: dict[str, list[float]] = {
                "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
            }
    
            epoch_bar = tqdm(range(num_epochs), desc=f"Fold {fold + 1}", leave=False)

            for _epoch in epoch_bar:
                # Training step
                train_loss, train_acc = self._train_step(train_loader, optimizer, criterion)

                # Validation step
                val_loss, val_acc = self.evaluate(val_loader, criterion)

                # Store in fold history
                fold_history["train_loss"].append(train_loss)
                fold_history["train_acc"].append(train_acc)
                fold_history["val_loss"].append(val_loss)
                fold_history["val_acc"].append(val_acc)
                
                # Store in global history
                global_history["train_loss"].append(train_loss)
                global_history["train_acc"].append(train_acc)
                global_history["val_loss"].append(val_loss)
                global_history["val_acc"].append(val_acc)

                epoch_bar.set_postfix(
                    train_loss=f'{train_loss:.4f}',
                    train_acc=f'{train_acc:.2f}%',
                    val_loss=f'{val_loss:.4f}',
                    val_acc=f'{val_acc:.2f}%'
                )

            # Average metrics over epochs for this fold and store
            for key in fold_averaged_history:
                fold_avg = sum(fold_history[key]) / len(fold_history[key])
                fold_averaged_history[key].append(fold_avg)
            
            # Print fold summary
            print(f"\nFold {fold + 1} Summary:")
            print(f"  Avg Train Loss: {fold_averaged_history['train_loss'][-1]:.4f}")
            print(f"  Avg Train Acc:  {fold_averaged_history['train_acc'][-1]:.2f}%")
            print(f"  Avg Val Loss:   {fold_averaged_history['val_loss'][-1]:.4f}")
            print(f"  Avg Val Acc:    {fold_averaged_history['val_acc'][-1]:.2f}%")

        # Average results across all folds
        averaged_history = {
            k: sum(v) / len(v) for k, v in fold_averaged_history.items()
        }

        print("\n===== Cross-validation results =====")
        print(f"Avg Train Loss: {averaged_history['train_loss']:.4f} \t Max Val Acc: {max(fold_averaged_history['val_acc']):.2f}%")
        print(f"Avg Train Acc:  {averaged_history['train_acc']:.2f}% \t Min Val Loss: {min(fold_averaged_history['val_loss']):.4f}")
        print(f"Avg Val Loss:   {averaged_history['val_loss']:.4f}   \t Min Val Loss: {min(fold_averaged_history['val_loss']):.4f}")
        print(f"Avg Val Acc:    {averaged_history['val_acc']:.2f}%   \t Max Val Acc: {max(fold_averaged_history['val_acc']):.2f}%")

        return global_history, averaged_history

    def _train_step(
        self,
        dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
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
        self, dataloader: DataLoader, criterion: nn.Module  # pyright: ignore[reportMissingTypeArgument]
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
