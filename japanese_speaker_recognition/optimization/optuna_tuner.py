from pathlib import Path
from typing import Any, Literal

import optuna
from optuna import Study, Trial
from torch import Tensor
import numpy as np

from japanese_speaker_recognition.models.HAIKU import HAIKU
from utils.utils import heading


class OptunaTuner:
    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        base_config: dict[str, Any],
        n_trials: int = 50,
        study_name: str = "HAIKU_speaker_recognition",
        seed: int = 42,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.base_config = base_config
        self.n_trials = n_trials
        self.study_name = study_name
        self.seed = seed
        self.study: Study | None = None

    @staticmethod
    def _tuning_ranges_from_config(
        config: dict[str, Any]
        ) -> dict[str, tuple[float, float] | list[int]]:
        """requires only the ranges part of the yaml config"""
        learning_rate_raw = list[float](config.get("LEARNING_RATE", [1e-5, 1e-2]))
        dropout_raw = list[float](config.get("DROPOUT", [0.1, 0.5]))
        conv_channels = list[int](config.get("CONV_CHANNELS", [64, 128, 256]))
        hidden_dim = list[int](config.get("HIDDEN_DIM", [32, 64, 128]))
        kernel_size = list[int](config.get("KERNEL_SIZE", [3, 5, 7]))
        batch_size = list[int](config.get("BATCH_SIZE", [16, 32, 64]))

        # Convert 2-element lists to tuples for ranges
        learning_rate = (float(learning_rate_raw[0]), float(learning_rate_raw[1]))
        dropout = (float(dropout_raw[0]), float(dropout_raw[1]))

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
        param_ranges_config = self.base_config.get("OPTUNA", {}).get("RANGES", {})
        suggested_params = self._tuning_ranges_from_config(param_ranges_config)

        learning_rate_low = suggested_params["LEARNING_RATE"][0]
        learning_rate_high = suggested_params["LEARNING_RATE"][1]

        dropout_low = float(suggested_params["DROPOUT"][0])
        dropout_high = float(suggested_params["DROPOUT"][1])

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
        ) -> dict[str, int | float | str]:
        """Creates the model config from the suggested hyperparameters."""
        model_config = self.base_config.get("MODEL", {})

        return {
            "DROPOUT": suggested_params.get("DROPOUT", 0.3),
            "EMBEDDING_DIM": model_config.get("embedding_dim", 64),
            "KERNEL_SIZE": suggested_params.get("KERNEL_SIZE", 5),
            "CONV_CHANNELS": suggested_params.get("CONV_CHANNELS", 128),
            "HIDDEN_DIM": suggested_params.get("HIDDEN_DIM", 64),
            "INPUT_CHANNELS": model_config.get("INPUT_CHANNELS", 12),
            "NUM_CLASSES": model_config.get("NUM_CLASSES", 9),
            "DEVICE": model_config.get("DEVICE", "cpu"),
        }

    def objective(self, trial: Trial) -> float:
        suggested_params = self._suggest_hyperparameters_from_config_ranges(trial)
        model_config = self._create_model_config(suggested_params)

        model = HAIKU._from_config(model_config)

        num_epochs = self.base_config.get("MODEL", {}).get("NUM_EPOCHS", 10)
        history = model.train_model(
            x_train=self.x_train,
            y_train=self.y_train,
            learning_rate=suggested_params["LEARNING_RATE"],
            num_epochs=num_epochs,
            batch_size=suggested_params["BATCH_SIZE"],
            k_folds=self.base_config.get("MODEL", {}).get("K_FOLDS", 5),
            seed=self.seed,
        )

        best_val_acc = np.mean(history["val_acc"])
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        return best_val_acc

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
        ) -> Study:
        heading("Starting Hyperparameter Optimization")

        self.study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            # show_progress_bar=show_progress_bar,
        )

        self._print_results()
        return self.study

    def get_best_config(self) -> dict[str, int | float | str]:
        if self.study is None:
            raise ValueError("Must run optimize() first.")
        
        best_params = self.study.best_params
        model_section = self.base_config.get("MODEL", {})
        return {
            "DROPOUT": best_params["DROPOUT"],
            "EMBEDDING_DIM": model_section["EMBEDDING_DIM"],
            "KERNEL_SIZE": best_params["KERNEL_SIZE"],
            "CONV_CHANNELS": best_params["CONV_CHANNELS"],
            "HIDDEN_DIM": best_params["HIDDEN_DIM"],
            "INPUT_CHANNELS": model_section.get("INPUT_CHANNELS", 12),
            "NUM_CLASSES": model_section.get("NUM_CLASSES", 9),
            "DEVICE": model_section.get("DEVICE", "cpu"),
            "LEARNING_RATE": best_params["LEARNING_RATE"],
            "BATCH_SIZE": best_params["BATCH_SIZE"],
            "NUM_EPOCHS": model_section.get("NUM_EPOCHS", 100),
        }

    def save_plots(self, output_dir: str | Path = ".") -> None:
        if self.study is None:
            print("No study has been run.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )

            # Optimization history
            fig = plot_optimization_history(self.study)
            plt.tight_layout()
            plt.savefig(output_dir / "optuna_optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Parameter importances
            fig = plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(output_dir / "optuna_param_importances.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study)
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
