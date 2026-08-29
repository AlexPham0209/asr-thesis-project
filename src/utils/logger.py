import os
from transformers import TrainerState
from transformers import TrainerCallback
import logging
from transformers.utils import logging as hf_logging
import sys


class CustomLoggingCallback(TrainerCallback):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_step = state.global_step
        metrics = " - ".join([f"{key}: {value:.4f}" for key, value in logs.items()])

        if self.logger is not None:
            self.logger.info(f"step {current_step} - {metrics}")


def initialize_loggers(cfg, timestamp):
    logging_directory = cfg.logging_directory
    os.makedirs(logging_directory, exist_ok=True)

    # Common log formatter
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    # Creating subfolder for current run
    run_directory = os.path.join(logging_directory, timestamp)
    os.makedirs(run_directory, exist_ok=True)

    # Screen/Console Handler (Attached to root so everything prints to stdout)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(file_formatter)

    # Root Logger Setup (Captures everything)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(screen_handler)

    root_file_handler = logging.FileHandler(
        os.path.join(run_directory, "all.log"), mode="w"
    )
    root_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(root_file_handler)

    # Application Logger Setup (Isolates your app's code logs via "finetuning")
    app_logger = logging.getLogger("finetuning")
    app_file_handler = logging.FileHandler(
        os.path.join(run_directory, "app.log"), mode="w"
    )
    app_file_handler.setFormatter(file_formatter)
    app_logger.addHandler(app_file_handler)
    app_logger.propagate = False

    # Hugging Face Logger Setup (Isolates Hugging Face transformers logs)
    hf_logger_instance = hf_logging.get_logger("transformers")
    hf_file_handler = logging.FileHandler(
        os.path.join(run_directory, "hf.log"), mode="w"
    )
    hf_file_handler.setFormatter(file_formatter)
    hf_logger_instance.addHandler(hf_file_handler)

    hf_logging.set_verbosity_info()
