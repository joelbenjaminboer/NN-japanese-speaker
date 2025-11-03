"""
A factory and dataclass file for the configuration settings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from typing_extensions import override


@dataclass
class AugmentationStep:
    """Configuration for a single augmentation step."""
    type: str
    noise_factor: float | None = None
    p: float | None = None
    scale_range: list[float] | None = None
    max_mask_percentage: float | None = None

    @override
    def __str__(self) -> str:
        parts = [f"Type: {self.type}"]
        if self.noise_factor is not None:
            parts.append(f"noise_factor={self.noise_factor}")
        if self.p is not None:
            parts.append(f"p={self.p}")
        if self.scale_range is not None:
            parts.append(f"scale_range={self.scale_range}")
        if self.max_mask_percentage is not None:
            parts.append(f"max_mask_percentage={self.max_mask_percentage}")
        return f"AugmentationStep({', '.join(parts)})"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugmentationStep":
        """Create AugmentationStep from dictionary."""
        return cls(
            type=data['type'],
            noise_factor=data.get('noise_factor'),
            p=data.get('p'),
            scale_range=data.get('scale_range'),
            max_mask_percentage=data.get('max_mask_percentage')
        )


@dataclass
class Augmentation:
    """Configuration for data augmentation."""
    enabled: bool
    repeats: int
    aug_dir: Path
    seed: int
    steps: list[AugmentationStep]

    @override
    def __str__(self) -> str:
        steps_str = "\n    ".join(str(step) for step in self.steps)
        return (
            f"Augmentation:\n"
            f"  Enabled: {self.enabled}\n"
            f"  Repeats: {self.repeats}\n"
            f"  Aug Dir: {self.aug_dir}\n"
            f"  Seed: {self.seed}\n"
            f"  Steps:\n    {steps_str}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Augmentation":
        """Create Augmentation from dictionary."""
        return cls(
            enabled=data['AUGMENT'],
            repeats=data['REPEATS'],
            aug_dir=Path(data['AUG_FILE']),
            seed=data['SEED'],
            steps=[AugmentationStep.from_dict(step) for step in data['STEPS']]
        )


@dataclass
class Embedding:
    """Configuration for embedding settings."""
    model: str
    dimension: int
    pre_precision: int
    batch_size: int
    output_dir: Path
    key: str

    @override
    def __str__(self) -> str:
        return (
            f"Embedding:\n"
            f"  Model: {self.model}\n"
            f"  Dimension: {self.dimension}\n"
            f"  Pre-precision: {self.pre_precision}\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Output Dir: {self.output_dir}\n"
            f"  Key: {self.key}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Embedding":
        """Create Embedding from dictionary."""
        return cls(
            model=data['MODEL'],
            dimension=data['DIMENSION'],
            pre_precision=data['PRE_PRECISION'],
            batch_size=data['BATCH_SIZE'],
            output_dir=Path(data['OUTPUT_FILE']),
            key=data['KEY'],
        )


@dataclass
class InputDirs:
    """Configuration for input directories."""
    train_file_dir: Path
    test_file_dir: Path

    @override
    def __str__(self) -> str:
        return (
            f"InputDirs:\n"
            f"  Train File: {self.train_file_dir}\n"
            f"  Test File: {self.test_file_dir}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputDirs":
        """Create InputDirs from dictionary."""
        return cls(
            train_file_dir=Path(data['TRAIN_FILE']),
            test_file_dir=Path(data['TEST_FILE'])
        )


@dataclass
class OutputDirs:
    """Configuration for output directories."""
    processed_file_dir: Path
    figures_dir: Path
    model_dir: Path

    @override
    def __str__(self) -> str:
        return (
            f"OutputDirs:\n"
            f"  Processed File: {self.processed_file_dir}\n"
            f"  Figures: {self.figures_dir}"
            f"  Model Dir: {self.model_dir}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputDirs":
        """Create OutputDirs from dictionary."""
        for key, value in data.items():
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            
        return cls(
            processed_file_dir=Path(data['PROCESSED']),
            figures_dir=Path(data['FIGURES']),
            model_dir=Path(data['MODELS'])
        )


@dataclass
class OptunaRanges:
    """Configuration for Optuna hyperparameter ranges."""
    learning_rate: list[float]
    dropout: list[float]
    dropout_mlp: list[float]
    conv_channels: list[int]
    hidden_dim: list[int]
    kernel_size: list[int]
    batch_size: list[int]

    @override
    def __str__(self) -> str:
        return (
            f"OptunaRanges:\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Dropout MLP: {self.dropout_mlp}\n"
            f"  Conv Channels: {self.conv_channels}\n"
            f"  Hidden Dim: {self.hidden_dim}\n"
            f"  Kernel Size: {self.kernel_size}\n"
            f"  Batch Size: {self.batch_size}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptunaRanges":
        """Create OptunaRanges from dictionary."""
        return cls(
            learning_rate=data['LEARNING_RATE'],
            dropout=data['DROPOUT'],
            dropout_mlp=data['DROPOUT_MLP'],
            conv_channels=data['CONV_CHANNELS'],
            hidden_dim=data['HIDDEN_DIM'],
            kernel_size=data['KERNEL_SIZE'],
            batch_size=data['BATCH_SIZE']
        )


@dataclass
class Optuna:
    """Configuration for Optuna hyperparameter optimization."""
    enabled: bool
    show_progress_bar: bool
    n_trials: int
    study_name: str
    storage_url: str
    figures_dir: Path
    study_dir: Path
    best_config_dir: Path
    ranges: OptunaRanges

    @override
    def __str__(self) -> str:
        return (
            f"Optuna:\n"
            f"  Enabled: {self.enabled}\n"
            f"  Show Progress Bar: {self.show_progress_bar}\n"
            f"  N Trials: {self.n_trials}\n"
            f"  Study Name: {self.study_name}\n"
            f"  Storage URL: {self.storage_url}\n"
            f"  Figures Dir: {self.figures_dir}\n"
            f"  Study Dir: {self.study_dir}\n"
            f"  Best Config Dir: {self.best_config_dir}\n"
            f"  {str(self.ranges).replace(chr(10), chr(10) + '  ')}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Optuna":
        """Create Optuna from dictionary."""
        return cls(
            enabled=data['ENABLED'],
            show_progress_bar=data['SHOW_PROGRESS_BAR'],
            n_trials=data['N_TRIALS'],
            study_name=data['STUDY_NAME'],
            storage_url=data.get('STORAGE_URL', 'sqlite:///optuna_study.db'),
            figures_dir=Path(data['FIGURES_DIR']),
            study_dir=Path(data['STUDY_DIR']),
            best_config_dir=Path(data['BEST_CONFIG_DIR']),
            ranges=OptunaRanges.from_dict(data['RANGES'])
        )


@dataclass
class Model:
    """Configuration for the neural network model."""
    load_best_config: bool
    num_classes: int
    embedding_dim: int
    kernel_size: int
    conv_channels: int
    dropout: float
    dropout_mlp: float
    input_channels: int
    hidden_dim: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    k_folds: int
    num_workers: int
    pin_memory: bool
    device: Literal["cuda", "cpu", "auto"]

    @override
    def __str__(self) -> str:
        return (
            f"Model:\n"
            f"  Load Best Config: {self.load_best_config}\n"
            f"  Num Classes: {self.num_classes}\n"
            f"  Embedding Dim: {self.embedding_dim}\n"
            f"  Kernel Size: {self.kernel_size}\n"
            f"  Conv Channels: {self.conv_channels}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Dropout MLP: {self.dropout_mlp}\n"
            f"  Input Channels: {self.input_channels}\n"
            f"  Hidden Dim: {self.hidden_dim}\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Num Epochs: {self.num_epochs}\n"
            f"  K-Folds: {self.k_folds}\n"
            f"  Num Workers: {self.num_workers}\n"
            f"  Pin Memory: {self.pin_memory}\n"
            f"  Device: {self.device}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        """Create Model from dictionary."""
        print(yaml.dump(data))
        return cls(
            load_best_config=data['LOAD_BEST_CONFIG'],
            num_classes=data['NUM_CLASSES'],
            embedding_dim=data['EMBEDDING_DIM'],
            kernel_size=data['KERNEL_SIZE'],
            conv_channels=data['CONV_CHANNELS'],
            dropout=data['DROPOUT'],
            dropout_mlp=data['DROPOUT_MLP'],
            input_channels=data['INPUT_CHANNELS'],
            hidden_dim=data['HIDDEN_DIM'],
            learning_rate=data['LEARNING_RATE'],
            batch_size=data['BATCH_SIZE'],
            num_epochs=data['NUM_EPOCHS'],
            k_folds=data['K_FOLDS'],
            num_workers=data['NUM_WORKERS'],
            pin_memory=data['PIN_MEMORY'],
            device=data['DEVICE']
        )


@dataclass
class Config:
    """Main configuration class containing all settings."""
    data_url: str
    seed: int
    max_length: int
    n_features: int
    augmentation: Augmentation
    embedding: Embedding
    input_dirs: InputDirs
    output_dirs: OutputDirs
    model: Model
    optuna: Optuna

    @override
    def __str__(self) -> str:
        sections = [
            "Configuration Settings",
            "=" * 50,
            f"Data URL: {self.data_url}",
            f"Seed: {self.seed}",
            f"Max Length: {self.max_length}",
            f"N Features: {self.n_features}",
            "",
            str(self.augmentation),
            "",
            str(self.embedding),
            "",
            str(self.input_dirs),
            "",
            str(self.output_dirs),
            "",
            str(self.model),
            "",
            str(self.optuna)
        ]
        return "\n".join(sections)

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config/config.yaml") -> "Config":
        """
        Load and parse the YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config object with all settings parsed and validated
        """
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
        
        return cls.from_dict(yaml_data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            Config object with all settings parsed
        """
        return cls(
            data_url=data['DATA_URL'],
            seed=data['SEED'],
            max_length=data['MAX_LEN'],
            n_features=data['N_FEATURES'],
            augmentation=Augmentation.from_dict(data['AUGMENTATION']),
            embedding=Embedding.from_dict(data['EMBEDDING']),
            input_dirs=InputDirs.from_dict(data['INPUT_DIRS']),
            output_dirs=OutputDirs.from_dict(data['OUTPUT_DIRS']),
            model=Model.from_dict(data['MODEL']),
            optuna=Optuna.from_dict(data['OPTUNA'])
        )