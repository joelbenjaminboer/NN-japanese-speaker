from typing import Self

import optuna
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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns

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
        dropout_mlp: float = 0.0,
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
        temp = dropout_mlp

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
            nn.Dropout(dropout_mlp),
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
            dropout_mlp=model_cfg.dropout_mlp,
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
        seed: int = 42,
        trial = None  # optuna.Trial for pruning support
        ) -> tuple[dict[str, list[float]], dict[str, float], dict[str, list[float]]]:
        """Train the model using K-Fold cross-validation and return averaged training history.
        
        Returns:
            tuple containing:
                - global_history: all epoch metrics across all folds
                - averaged_history: single averaged value per metric across all folds
                - fold_averaged_history: average metrics for each fold (list with one value per fold)
        """

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
            
            # Report intermediate value for pruning
            if trial is not None:
                trial.report(fold_averaged_history['val_acc'][-1], fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Average results across all folds
        averaged_history = {
            k: sum(v) / len(v) for k, v in fold_averaged_history.items()
        }

        print("\n===== Cross-validation results =====")
        print(f"Avg Train Loss: {averaged_history['train_loss']:.4f} \t Max Val Acc: {max(fold_averaged_history['val_acc']):.2f}%")
        print(f"Avg Train Acc:  {averaged_history['train_acc']:.2f}% \t Min Val Loss: {min(fold_averaged_history['val_loss']):.4f}")
        print(f"Avg Val Loss:   {averaged_history['val_loss']:.4f}   \t Min Val Loss: {min(fold_averaged_history['val_loss']):.4f}")
        print(f"Avg Val Acc:    {averaged_history['val_acc']:.2f}%   \t Max Val Acc: {max(fold_averaged_history['val_acc']):.2f}%")

        return global_history, averaged_history, fold_averaged_history

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
    
    def save_onnx(
        self,
        file_path: str = "haiku_model.onnx",
        input_shape: tuple[int, int, int] | None = None,
        opset_version: int = 17,
        dynamic_batch: bool = True,
    ) -> None:
        """
        Export the trained model to ONNX format.

        Args:
            file_path: Output ONNX file path.
            input_shape: Example input shape (default: (1, 12, embedding_dim)).
            opset_version: ONNX opset version (default 17).
            dynamic_batch: Whether to allow dynamic batch sizes.
        """
        self.eval()  # Switch to eval mode
        input_shape = input_shape or (1, 12, self.embedding_dim)

        # Create dummy input tensor
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Dynamic axes allows variable batch sizes at inference
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_batch else None

        print(f"Exporting HAIKU to ONNX → {file_path}")
        print(f"Input shape: {tuple(dummy_input.shape)} | Opset: {opset_version}")

        torch.onnx.export(
            self,
            dummy_input,
            file_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        print(f"Model successfully exported to {file_path}")
        
    def save_confusion_matrix(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        file_path: str = "internal_confusion_matrix.png",
        batch_size: int = 32,
        class_names: list[str] | None = None,
    ) -> None:
        """
        Generate and save confusion matrices showing internal layer predictions.
        
        This visualizes how well different stages of the model can classify the data:
        1. After convolution layer (CNN features)
        2. After global pooling
        3. After first linear layer (hidden representation)
        4. Final output (full model)
        
        Args:
            x_data: Input features tensor [N, 12, embedding_dim]
            y_data: True labels tensor [N]
            file_path: Output file path for the confusion matrix grid
            batch_size: Batch size for evaluation
            class_names: List of class names for axis labels
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from sklearn.linear_model import LogisticRegression
        import seaborn as sns
        
        self.eval()
        self.to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Storage for internal representations
        conv_features = []
        pooled_features = []
        hidden_features = []
        final_logits = []
        all_labels = []
        
        print("Extracting internal representations...")
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader, desc="Processing"):
                batch_x = batch_x.to(self.device)
                
                # 1. After convolution
                conv_out = self.convolution(batch_x)
                conv_features.append(conv_out.cpu())
                
                # 2. After pooling
                pooled_out = self.global_pool(conv_out)
                pooled_features.append(pooled_out.squeeze(-1).cpu())
                
                # 3. After first linear layer (hidden)
                flattened = pooled_out.view(pooled_out.size(0), -1)
                hidden_out = self.perceptron[1](flattened)  # First linear layer
                hidden_out = self.perceptron[2](hidden_out)  # GELU
                hidden_features.append(hidden_out.cpu())
                
                # 4. Final output
                final_out = self(batch_x)
                final_logits.append(final_out.cpu())
                
                all_labels.append(batch_y.cpu())
        
        # Concatenate all batches
        conv_features = torch.cat(conv_features, dim=0)
        pooled_features = torch.cat(pooled_features, dim=0)
        hidden_features = torch.cat(hidden_features, dim=0)
        final_logits = torch.cat(final_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Train simple classifiers on internal representations
        print("Training classifiers on internal representations...")
        
        # Flatten conv features for classifier
        conv_flat = conv_features.view(conv_features.size(0), -1).numpy()
        pooled_flat = pooled_features.numpy()
        hidden_flat = hidden_features.numpy()
        
        # Train logistic regression on each representation
        clf_conv = LogisticRegression(max_iter=1000, random_state=42)
        clf_conv.fit(conv_flat, all_labels)
        pred_conv = clf_conv.predict(conv_flat)
        
        clf_pooled = LogisticRegression(max_iter=1000, random_state=42)
        clf_pooled.fit(pooled_flat, all_labels)
        pred_pooled = clf_pooled.predict(pooled_flat)
        
        clf_hidden = LogisticRegression(max_iter=1000, random_state=42)
        clf_hidden.fit(hidden_flat, all_labels)
        pred_hidden = clf_hidden.predict(hidden_flat)
        
        # Final model predictions
        _, pred_final = torch.max(final_logits, 1)
        pred_final = pred_final.numpy()
        
        # Create confusion matrices
        cm_conv = confusion_matrix(all_labels, pred_conv)
        cm_pooled = confusion_matrix(all_labels, pred_pooled)
        cm_hidden = confusion_matrix(all_labels, pred_hidden)
        cm_final = confusion_matrix(all_labels, pred_final)
        
        # Normalize
        cm_conv = cm_conv.astype('float') / cm_conv.sum(axis=1)[:, np.newaxis]
        cm_pooled = cm_pooled.astype('float') / cm_pooled.sum(axis=1)[:, np.newaxis]
        cm_hidden = cm_hidden.astype('float') / cm_hidden.sum(axis=1)[:, np.newaxis]
        cm_final = cm_final.astype('float') / cm_final.sum(axis=1)[:, np.newaxis]
        
        # Set up class names
        if class_names is None:
            class_names = [str(i) for i in range(cm_final.shape[0])]
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        matrices = [
            (cm_conv, "After Convolution Layer", axes[0, 0]),
            (cm_pooled, "After Global Pooling", axes[0, 1]),
            (cm_hidden, "After Hidden Layer", axes[1, 0]),
            (cm_final, "Final Output", axes[1, 1])
        ]
        
        accuracies = []
        
        for cm, title, ax in matrices:
            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={'label': 'Proportion'}
            )
            
            acc = 100 * np.trace(cm) / cm.shape[0]
            accuracies.append(acc)
            ax.set_title(f"{title}\nAccuracy: {acc:.2f}%", fontsize=12, pad=10)
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        plt.suptitle('Internal Model Confusion Matrices', fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"\nInternal confusion matrices saved to {file_path}")
        print("\nAccuracy at each stage:")
        print(f"  Conv Layer:    {accuracies[0]:.2f}%")
        print(f"  Pooled:        {accuracies[1]:.2f}%")
        print(f"  Hidden Layer:  {accuracies[2]:.2f}%")
        print(f"  Final Output:  {accuracies[3]:.2f}%")
        
        plt.close()
        
