import os

from matplotlib import pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_intermediate_values,
    plot_optimization_history,
    plot_param_importances,
)


def compute_objective(metrics):
    # Optuna will minimize the evaluation loss by default (or use "eval_wer" for WER minimization)
    return metrics["eval_cer"]


def hp_space(trial):
    """Defines the search space for Optuna trials."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [1, 2, 4]
        ),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine", "constant"]
        ),
    }


def create_hyperparameter_diagrams(
    name: str, model_directory: str, studies_directory: str
):
    # Load the study from RDB storage
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{studies_directory}/{name}_optuna_trials.db"
    )

    study = optuna.load_study(study_name=f"{name}_optuna_study", storage=storage)

    # Plot optimization history
    ax1 = plot_optimization_history(study)
    ax1.figure.savefig(os.path.join(model_directory, "optimization_history.png"))

    # Plot intermediate values (if using pruning and intermediate reports)
    ax2 = plot_intermediate_values(study)
    ax2.figure.savefig(os.path.join(model_directory, "intermediate_values.png"))

    # Plot parameter importances
    ax3 = plot_param_importances(study)
    ax3.figure.savefig(os.path.join(model_directory, "param_importances.png"))
