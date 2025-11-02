

"""
Augmentation Experiment Pipeline

Strategy:
1. Create modular augmentation configurations
2. Pre-embed datasets with different augmentation settings
3. Use Optuna to jointly optimize:
   - Augmentation parameters (which augmentations, probabilities, strengths)
   - Model hyperparameters (learning rate, dropout, architecture)
4. Early stopping via MedianPruner to terminate unpromising trials

Design rationale:
- Augmentations are applied BEFORE embedding to maximize efficiency
- Each trial tests a unique augmentation+model combination
- MedianPruner chosen for its effectiveness with K-fold CV metrics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config.config import AugmentationStep, Config
from japanese_speaker_recognition.data_augmentation import (
    AddGaussianNoise,
    AugmentationPipeline,
    FrequencyMasking,
    Probabilistic,
    RandomScaling,
    TimeMasking,
)
from japanese_speaker_recognition.dataset import JapaneseVowelsDataset
from japanese_speaker_recognition.models.HAIKU import HAIKU
from utils.utils import heading


@dataclass
class AugmentationConfig:
    """Modular augmentation configuration for experiments."""
    gaussian_noise_enabled: bool = False
    gaussian_noise_factor: float = 0.001
    gaussian_noise_p: float = 1.0
    
    random_scaling_enabled: bool = False
    random_scaling_range: tuple[float, float] = (0.95, 1.05)
    random_scaling_p: float = 1.0
    
    time_masking_enabled: bool = False
    time_masking_percentage: float = 0.01
    time_masking_p: float = 1.0
    
    frequency_masking_enabled: bool = False
    frequency_masking_percentage: float = 0.01
    frequency_masking_p: float = 1.0
    
    repeats: int = 2
    seed: int = 42

    def to_augmentation_steps(self) -> list[AugmentationStep]:
        """Convert config to list of augmentation steps."""
        steps = []
        
        if self.gaussian_noise_enabled:
            steps.append(AugmentationStep(
                type="gaussian_noise",
                noise_factor=self.gaussian_noise_factor,
                p=self.gaussian_noise_p
            ))
        
        if self.random_scaling_enabled:
            steps.append(AugmentationStep(
                type="random_scaling",
                scale_range=list(self.random_scaling_range),
                p=self.random_scaling_p
            ))
        
        if self.time_masking_enabled:
            steps.append(AugmentationStep(
                type="time_masking",
                max_mask_percentage=self.time_masking_percentage,
                p=self.time_masking_p
            ))
        
        if self.frequency_masking_enabled:
            steps.append(AugmentationStep(
                type="frequency_masking",
                max_mask_percentage=self.frequency_masking_percentage,
                p=self.frequency_masking_p
            ))
        
        return steps


class AugmentationExperiment:
    """
    Orchestrates augmentation experiments with Optuna optimization.
    
    Uses MedianPruner for early stopping - prunes trials whose intermediate
    values are worse than the median of all trials at the same step.
    This is effective for K-fold CV where we get intermediate results per fold.
    """
    
    def __init__(
        self,
        cfg: Config,
        device: str = "auto",
        n_trials: int = 50,
        study_name: str = "augmentation_experiment"
    ):
        self.cfg = cfg
        self.device = self._get_device(device)
        self.n_trials = n_trials
        self.study_name = study_name
        
        # Storage for embedded datasets
        self.dataset_cache: dict[str, dict[str, Any]] = {}
    
    @staticmethod
    def _get_device(device: str) -> str:
        """Resolve device configuration."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return "cpu"
        return device
    
    def _create_augmentation_pipeline(
        self, aug_config: AugmentationConfig
    ) -> AugmentationPipeline:
        """Create augmentation pipeline from config."""
        steps = aug_config.to_augmentation_steps()
        
        # Convert AugmentationStep configs to actual Probabilistic transforms
        probabilistic_steps = []
        for step in steps:
            transform = None
            p = step.p if step.p is not None else 1.0
            
            if step.type == "gaussian_noise":
                transform = AddGaussianNoise(noise_factor=step.noise_factor)
            elif step.type == "random_scaling":
                transform = RandomScaling(scale_range=tuple(step.scale_range))
            elif step.type == "time_masking":
                transform = TimeMasking(max_mask_percentage=step.max_mask_percentage)
            elif step.type == "frequency_masking":
                transform = FrequencyMasking(max_mask_percentage=step.max_mask_percentage)
            else:
                raise ValueError(f"Unknown augmentation type: {step.type}")
            
            probabilistic_steps.append(Probabilistic(transform=transform, p=p))
        
        return AugmentationPipeline(steps=probabilistic_steps, seed=aug_config.seed)
    
    def _get_or_create_dataset(
        self, aug_config: AugmentationConfig, cache_key: str
    ) -> dict[str, Any]:
        """
        Get cached dataset or create new one with specified augmentations.
        
        Cache key strategy: hash augmentation parameters to avoid redundant embeddings.
        """
        if cache_key in self.dataset_cache:
            print(f"Using cached dataset: {cache_key}")
            return self.dataset_cache[cache_key]
        
        print(f"Creating new dataset with augmentation config: {cache_key}")
        
        # Create augmentation pipeline
        augmenter = None
        if any([
            aug_config.gaussian_noise_enabled,
            aug_config.random_scaling_enabled,
            aug_config.time_masking_enabled,
            aug_config.frequency_masking_enabled
        ]):
            augmenter = self._create_augmentation_pipeline(aug_config)
        
        # Update config for this experiment
        temp_cfg = self.cfg # TODO: deep copy? no? 
        temp_cfg.augmentation.enabled = augmenter is not None
        temp_cfg.augmentation.repeats = aug_config.repeats
        
        # Create dataset with augmentation
        dataset = JapaneseVowelsDataset(
            cfg=temp_cfg,
            augmenter=augmenter,
            embedding_dim=self.cfg.embedding.dimension,
            embedding_model=self.cfg.embedding.model,
            embedidng_precision=self.cfg.embedding.pre_precision,
            device=self.device,
            key=cache_key
        )
        
        # Prepare dataset (download, augment, embed, save)
        data = dataset.prepare()
        
        # Cache for future use
        self.dataset_cache[cache_key] = data
        
        return data
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Optimizes both augmentation parameters and model hyperparameters jointly.
        Reports intermediate values per fold for pruning.
        """
        
        # ===== Suggest Augmentation Parameters =====
        aug_config = AugmentationConfig(
            gaussian_noise_enabled=trial.suggest_categorical("aug_gaussian_noise", [True, False]),
            gaussian_noise_factor=trial.suggest_float("aug_gaussian_noise_factor", 0.0001, 0.01, log=True),
            gaussian_noise_p=trial.suggest_float("aug_gaussian_noise_p", 0.3, 1.0),
            
            random_scaling_enabled=trial.suggest_categorical("aug_random_scaling", [True, False]),
            random_scaling_range=(
                trial.suggest_float("aug_scaling_low", 0.90, 0.98),
                trial.suggest_float("aug_scaling_high", 1.02, 1.10)
            ),
            random_scaling_p=trial.suggest_float("aug_random_scaling_p", 0.3, 1.0),
            
            time_masking_enabled=trial.suggest_categorical("aug_time_masking", [True, False]),
            time_masking_percentage=trial.suggest_float("aug_time_masking_pct", 0.005, 0.05),
            time_masking_p=trial.suggest_float("aug_time_masking_p", 0.3, 1.0),
            
            frequency_masking_enabled=trial.suggest_categorical("aug_freq_masking", [True, False]),
            frequency_masking_percentage=trial.suggest_float("aug_freq_masking_pct", 0.005, 0.05),
            frequency_masking_p=trial.suggest_float("aug_freq_masking_p", 0.3, 1.0),
            
            repeats=trial.suggest_int("aug_repeats", 1, 4),
            seed=self.cfg.seed
        )
        
        # ===== Suggest Model Hyperparameters =====
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        conv_channels = trial.suggest_categorical("conv_channels", [128, 256, 512])
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])  # H100-optimized
        
        # ===== Create Cache Key =====
        # Key includes all augmentation parameters to avoid embedding duplicates
        cache_key = (
            f"gn{int(aug_config.gaussian_noise_enabled)}_"
            f"{aug_config.gaussian_noise_factor:.4f}_"
            f"rs{int(aug_config.random_scaling_enabled)}_"
            f"{aug_config.random_scaling_range[0]:.2f}_{aug_config.random_scaling_range[1]:.2f}_"
            f"tm{int(aug_config.time_masking_enabled)}_"
            f"{aug_config.time_masking_percentage:.3f}_"
            f"fm{int(aug_config.frequency_masking_enabled)}_"
            f"{aug_config.frequency_masking_percentage:.3f}_"
            f"r{aug_config.repeats}"
        )
        
        # ===== Get or Create Dataset =====
        data = self._get_or_create_dataset(aug_config, cache_key)
        
        # Convert to tensors
        X_train = torch.tensor(np.array(data["X_train"]), dtype=torch.float32).to(self.device)
        y_train = torch.tensor(data["y_train"], dtype=torch.long).to(self.device)
        
        # ===== Create and Train Model =====
        model = HAIKU(
            num_classes=self.cfg.model.num_classes,
            dropout=dropout,
            embedding_dim=self.cfg.embedding.dimension,
            kernel_size=kernel_size,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
            input_channels=self.cfg.model.input_channels,
            device=self.device
        ).to(self.device)
        
        # Train with K-fold CV, report intermediate results for pruning
        _history, avg_history = model.train_model(
            x_train=X_train,
            y_train=y_train,
            learning_rate=learning_rate,
            num_epochs=self.cfg.model.num_epochs,
            batch_size=batch_size,
            k_folds=self.cfg.model.k_folds,
            num_workers=self.cfg.model.num_workers if self.device == "cpu" else 0,
            pin_memory=self.cfg.model.pin_memory if self.device == "cpu" else False,
            seed=self.cfg.seed
        )
        
        # Return average validation accuracy across all folds
        return avg_history["val_acc"]
    
    def run_optimization(self, storage_url: str | None = None) -> optuna.Study:
        """
        Run Optuna optimization with early stopping.
        
        MedianPruner: Prunes trials with intermediate values worse than median.
        Effective for K-fold CV where we have multiple intermediate metrics.
        
        n_startup_trials: Number of random trials before pruning kicks in (warmup).
        n_warmup_steps: Number of steps before pruner evaluates (wait for signal).
        interval_steps: Check every N steps (here every fold).
        """
        heading("Starting Augmentation + Model Optimization")
        
        # Storage for parallel/distributed optimization
        if storage_url is None:
            storage_url = "sqlite:///exp_augmentation_study.db"
        
        # MedianPruner: stops trials performing worse than median at same step
        pruner = MedianPruner(
            n_startup_trials=5,  # warmup: don't prune first 5 trials
            n_warmup_steps=2,    # wait for 2 folds before considering pruning
            interval_steps=1     # evaluate pruning after each fold
        )
        
        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            sampler=TPESampler(seed=self.cfg.seed),
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=self.cfg.optuna.show_progress_bar,
            n_jobs=1  # parallel jobs handled at SLURM/external level
        )
        
        # ===== Print Results =====
        heading("Optimization Complete")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation accuracy: {study.best_value:.2f}%")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # ===== Save Results =====
        self._save_study(study)
        self._save_plots(study)
        
        return study
    
    def _save_study(self, study: optuna.Study) -> None:
        """Save study object and best config."""
        import joblib
        
        study_dir = Path(self.cfg.optuna.study_dir)
        study_dir.mkdir(parents=True, exist_ok=True)
        
        study_path = study_dir / f"{self.study_name}.pkl"
        joblib.dump(study, study_path)
        print(f"\nStudy saved to: {study_path}")
        
        # Save best config as YAML
        best_config_dir = Path(self.cfg.optuna.best_config_dir)
        best_config_dir.mkdir(parents=True, exist_ok=True)
        
        import yaml
        best_config_path = best_config_dir / f"{self.study_name}_best_config.yaml"
        with open(best_config_path, 'w') as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
        print(f"Best config saved to: {best_config_path}")
    
    def _save_plots(self, study: optuna.Study) -> None:
        """Save optimization plots."""
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )
            
            figures_dir = Path(self.cfg.optuna.figures_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Optimization history
            fig = plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{self.study_name}_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter importances
            fig = plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{self.study_name}_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parallel coordinate plot
            fig = plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{self.study_name}_parallel.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nPlots saved to: {figures_dir}")
            
        except ImportError as e:
            print(f"\nWarning: Could not generate plots: {e}")


def main():
    # ===== Load Config =====
    heading("Config Settings")
    cfg = Config.from_yaml()
    print(cfg)
    
    # ===== Device Configuration =====
    heading("Device Configuration")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.model.device != device:
        print(f"Warning: Configured device {cfg.model.device} is not available. Using {device} instead.")
    else:
        print(f"Using device: {device}")
    
    # ===== Run Augmentation Experiment =====
    experiment = AugmentationExperiment(
        cfg=cfg,
        device=device,
        n_trials=cfg.optuna.n_trials,
        study_name="augmentation_optimization_v1"
    )
    
    study = experiment.run_optimization()
    
    heading("Experiment complete!")


if __name__ == "__main__":
    main() 