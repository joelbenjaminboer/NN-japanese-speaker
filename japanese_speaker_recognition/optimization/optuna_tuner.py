from pathlib import Path
from typing import Literal

import optuna
from optuna import Study, Trial
from torch import Tensor

from config.config import Config, Model, OptunaRanges
from japanese_speaker_recognition.models.HAIKU import HAIKU
from utils.utils import heading


class OptunaTuner:
    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        config: Config,
        n_trials: int = 50,
        study_name: str = "HAIKU_speaker_recognition",
        device: str = "cpu",
        seed: int = 42,
    ):
        self.x_train: Tensor = x_train
        self.y_train: Tensor = y_train
        self.config: Config = config
        self.n_trials: int = n_trials
        self.study_name: str = study_name
        self.seed: int = seed
        self.study: Study | None = None
        self.device: str = device

    @staticmethod
    def _tuning_ranges_from_config(
        optuna_ranges: OptunaRanges
        ) -> dict[str, tuple[float, float] | list[int]]:
        """requires only the ranges part of the yaml config"""
        learning_rate_raw: list[float] = optuna_ranges.learning_rate
        dropout_raw: list[float] = optuna_ranges.dropout
        conv_channels: list[int] = optuna_ranges.conv_channels
        hidden_dim: list[int] = optuna_ranges.hidden_dim
        kernel_size: list[int] = optuna_ranges.kernel_size
        batch_size: list[int] = optuna_ranges.batch_size

        # Convert 2-element lists to tuples for ranges
        learning_rate: tuple[float, float] = (
            float(learning_rate_raw[0]), float(learning_rate_raw[1])
            )
        dropout: tuple[float, float] = (float(dropout_raw[0]), float(dropout_raw[1]))

        return {
            "LEARNING_RATE": learning_rate,
            "DROPOUT": dropout,
            "CONV_CHANNELS": conv_channels,
            "HIDDEN_DIM": hidden_dim,
            "KERNEL_SIZE": kernel_size,
            "BATCH_SIZE": batch_size,
        }


    def _suggest_hyperparameters_from_config_ranges(self, trial: Trial) -> dict[str, int | float]:
        """Takes self.base_config and suggests hyperparameters."""
        param_ranges_config: OptunaRanges = self.config.optuna.ranges
        suggested_params = self._tuning_ranges_from_config(param_ranges_config)

        learning_rate_low: float = suggested_params["LEARNING_RATE"][0]
        learning_rate_high: float = suggested_params["LEARNING_RATE"][1]

        dropout_low: float = suggested_params["DROPOUT"][0]
        dropout_high: float = suggested_params["DROPOUT"][1]

        return {
            "LEARNING_RATE": trial.suggest_float(
                "LEARNING_RATE", learning_rate_low, learning_rate_high, log=True
            ),
            "DROPOUT": trial.suggest_float(
                "DROPOUT", dropout_low, dropout_high
            ),
            "CONV_CHANNELS": trial.suggest_categorical(
                "CONV_CHANNELS", suggested_params["CONV_CHANNELS"]
            ),
            "HIDDEN_DIM": trial.suggest_categorical(
                "HIDDEN_DIM", suggested_params["HIDDEN_DIM"]
            ),
            "KERNEL_SIZE": trial.suggest_categorical(
                "KERNEL_SIZE", suggested_params["KERNEL_SIZE"]
            ),
            "BATCH_SIZE": trial.suggest_categorical(
                "BATCH_SIZE", suggested_params["BATCH_SIZE"]
            ),
        }

    def _create_model_config(
        self,
        suggested_params: dict[str, int | float]
        ) -> Model:
        """Creates the model config from the suggested hyperparameters."""
        model_config: Model = self.config.model

        return Model(
            load_best_config=model_config.load_best_config,
            num_classes=model_config.num_classes,
            embedding_dim=model_config.embedding_dim,
            kernel_size=int(suggested_params.get("KERNEL_SIZE", 3)),
            conv_channels=int(suggested_params.get("CONV_CHANNELS", 32)),
            dropout=float(suggested_params.get("DROPOUT", 0.1)),
            input_channels=model_config.input_channels,
            hidden_dim=int(suggested_params.get("HIDDEN_DIM", 64)),
            learning_rate=float(suggested_params.get("LEARNING_RATE", 1e-4)),
            batch_size=int(suggested_params.get("BATCH_SIZE", 32)),
            num_epochs=model_config.num_epochs,
            k_folds=model_config.k_folds,
            num_workers=model_config.num_workers,
            pin_memory=model_config.pin_memory,
            device=model_config.device
        )

    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna hyperparameter optimization.
        
        Returns:
            """
        suggested_params = self._suggest_hyperparameters_from_config_ranges(trial)
        model_config = self._create_model_config(suggested_params)

        model = HAIKU._from_config(model_config).to(self.device)  # pyright: ignore[reportPrivateUsage]

        model_device = next(model.parameters()).device

        if model_device.type == 'cuda':
            num_workers = 0
            use_pin_memory = False
            print(f"Data on GPU - using num_workers=0, pin_memory=False")
        else:
            num_workers = self.config.model.num_workers
            use_pin_memory = model_config.pin_memory
            print(f"Data on CPU - using num_workers={num_workers}, pin_memory={use_pin_memory}")

        num_epochs = self.config.model.num_epochs
        history, avg_history = model.train_model(
            x_train=self.x_train,
            y_train=self.y_train,
            learning_rate=suggested_params["LEARNING_RATE"],
            num_epochs=num_epochs,
            batch_size=int(suggested_params["BATCH_SIZE"]),
            k_folds=self.config.model.k_folds,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            seed=self.seed,
        )

        max_val_acc = max(fold_history["val_acc"] for fold_history in history)
        # best_val_acc = avg_history["val_acc"]
        
        print(f"Max validation accuracy across folds: {max_val_acc:.4f}")
        print(f"Average validation accuracy: {avg_history['val_acc']:.4f}")
        return max_val_acc

    def _print_results(self) -> None:
        if self.study is None:
            print("No study has been run.")
            return

        heading("Optimization Complete")
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best validation accuracy: {self.study.best_value:.2f}%")
        print("\nBest hyperparameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")

    def optimize(
        self,
        direction: Literal["minimize", "maximize"] = "maximize",
        show_progress_bar: bool = True,
        storage_url: str | None = None,   # <--- NEW
    ) -> Study:
        heading("Starting Hyperparameter Optimization")

        # Use shared database for parallel jobs
        if storage_url is None:
            storage_url = "sqlite:///optuna_study.db"

        # Configure SQLite for NFS environments with high timeout
        # Note: WAL mode does NOT work on network filesystems (NFS) due to
        # shared memory requirements. Use default rollback journal with high timeout.
        if storage_url.startswith("sqlite"):
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={
                    "connect_args": {
                        "timeout": 120.0,  # High timeout for NFS lock contention
                        "check_same_thread": False,
                        "isolation_level": None,  # Autocommit mode
                    },
                    "pool_pre_ping": True,  # Verify connections before using
                    "pool_size": 1,  # Single connection per worker
                    "max_overflow": 0,  # No overflow connections
                },
            )
        else:
            storage = storage_url

        self.study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            storage=storage,  # <--- Use configured storage
            load_if_exists=True,  # <--- allows multiple workers
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
            n_jobs=1,  # Optuna parallelism handled at the SLURM job level
        )

        self._print_results()
        return self.study


    def get_best_config(self) -> dict[str, int | float | str]:
        if self.study is None:
            raise ValueError("Must run optimize() first.")
        
        best_params = self.study.best_params
        model_section = self.config.model
        return {
            "DROPOUT": best_params["DROPOUT"],
            "EMBEDDING_DIM": model_section.embedding_dim,
            "KERNEL_SIZE": best_params["KERNEL_SIZE"],
            "CONV_CHANNELS": best_params["CONV_CHANNELS"],
            "HIDDEN_DIM": best_params["HIDDEN_DIM"],
            "INPUT_CHANNELS": model_section.input_channels,
            "NUM_CLASSES": model_section.num_classes,
            "DEVICE": model_section.device,
            "LEARNING_RATE": best_params["LEARNING_RATE"],
            "BATCH_SIZE": best_params["BATCH_SIZE"],
            "NUM_EPOCHS": model_section.num_epochs,
            "LOAD_BEST_CONFIG": model_section.load_best_config,
            "K_FOLDS": model_section.k_folds,
        }

    def save_plots(self, output_dir: str | Path = ".") -> None:
        if self.study is None:
            print("No study has been run.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_param_importances,
            )

            # Optimization history
            _ = plot_optimization_history(self.study)
            plt.tight_layout()
            plt.savefig(
                output_dir / "optuna_optimization_history.png", 
                dpi=300, 
                bbox_inches='tight'
                )
            plt.close()

            # Parameter importances
            _ = plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(output_dir / "optuna_param_importances.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Parallel coordinate plot
            _ = plot_parallel_coordinate(self.study)
            plt.tight_layout()
            plt.savefig(output_dir / "optuna_parallel_coordinate.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nOptuna plots saved to {output_dir}:")
            print("  - optuna_optimization_history.png")
            print("  - optuna_param_importances.png")
            print("  - optuna_parallel_coordinate.png")

        except ImportError as e:
            print(f"\nError generating plots: {e}")
            print("Make sure matplotlib is installed:")
            print("  pip install matplotlib")

    def save_study(self, output_dir: str | Path = "optuna_study.pkl") -> None:
        if self.study is None:
            print("No study has been run.")
            return

        import joblib

        joblib.dump(self.study, output_dir)
        print(f"Study saved to {output_dir}")

    @staticmethod
    def load_study(input_dir: str | Path = "optuna_study.pkl") -> Study:
        import joblib

        return joblib.load(input_dir)
