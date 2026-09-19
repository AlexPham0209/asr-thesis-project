import os
from dataclasses import dataclass


@dataclass
class RunPaths:
    run_name: str
    study_name: str
    model_directory: str
    studies_directory: str
    study_storage: str


def resolve_run_paths(cfg, timestamp: str) -> RunPaths:
    """run_name -> models/<run_name>; study_name -> studies/<study_name>.
    Both default to model_name; use_timestamp appends the timestamp to both."""
    model_name = cfg.get("model_name", "model")
    run_name = cfg.get("run_name") or model_name
    study_name = cfg.get("study_name") or run_name
    if cfg.get("use_timestamp", False):
        run_name = f"{run_name}_{timestamp}"
        study_name = f"{study_name}_{timestamp}"

    model_directory = os.path.join(cfg.get("model_directory", "models"), run_name)
    studies_directory = os.path.join(cfg.get("studies_directory", "studies"), study_name)
    os.makedirs(studies_directory, exist_ok=True)

    return RunPaths(
        run_name=run_name,
        study_name=study_name,
        model_directory=model_directory,
        studies_directory=studies_directory,
        study_storage=f"sqlite:///{studies_directory}/{study_name}_optuna_trials.db",
    )
