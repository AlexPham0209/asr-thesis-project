from builtins import getattr
from datetime import datetime
import logging
import os
import sys
import time
from data.latex_metrics import LatexInContextMetrics
import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC
import numpy as np
from hydra.utils import instantiate
from datasets import load_dataset
from data.normalizer import create_latex_normalizer
from utils.logger import CustomLoggingCallback
from utils.metrics import create_metric

from data.data_collator import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)

import logging
from transformers.utils import logging as hf_logging
from peft import get_peft_model, LoraConfig
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_seq2seq_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator, timestamp
):
    model_path = os.path.join(
        cfg.model_directory, f"{cfg.get('model_name', 'model')}_{timestamp}"
    )
    training_args = Seq2SeqTrainingArguments(**cfg.training, output_dir=model_path)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model_init=model,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[CustomLoggingCallback(logger)],
    )

    return trainer


def create_ctc_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator, timestamp
):
    model_path = os.path.join(cfg.model_directory, model.config._name_or_path)
    training_args = TrainingArguments(**cfg.training, output_dir=model_path)

    trainer = Trainer(
        model_init=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[CustomLoggingCallback(logger)],
    )

    return trainer


def create_custom_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator
):
    pass


def compute_objective(metrics):
    # Optuna will minimize the evaluation loss by default (or use "eval_wer" for WER minimization)
    return metrics["eval_loss"]


def hp_space(trial):
    """Defines the search space for Optuna trials."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
    }


def inference(model, processor, normalizer, dataset, architecture):
    # Metrics
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    predictions = []
    labels = []
    rtfxs = []

    for sample in dataset.take(10):
        key = "input_values" if architecture == "ctc" else "input_features"
        input_features = sample[key]

        # Ensure tensor type
        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features)

        # Move inputs to the correct device and add a batch dimension
        input_features = input_features.unsqueeze(dim=0).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            if architecture == "ctc":
                logits = model(input_features).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            else:
                predicted_ids = model.generate(input_features=input_features)

        end_time = time.perf_counter()

        label_ids = sample["labels"]
        if not isinstance(label_ids, torch.Tensor):
            label_ids = torch.tensor(label_ids)

        # Replace -100 padding tokens with pad_token_id
        pad_token_id = (
            getattr(processor, "pad_token_id", None) or processor.tokenizer.pad_token_id
        )
        label_ids = torch.where(label_ids != -100, label_ids, pad_token_id)

        label_ids = label_ids.unsqueeze(0)

        # Decoding prediction and labels
        pred_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(
            label_ids, skip_special_tokens=True, group_tokens=False
        )

        audio_duration = sample["input_length"]
        processing_time = end_time - start_time
        rtfx = audio_duration / processing_time

        predictions.extend(pred_str)
        labels.extend(label_str)
        rtfxs.append(rtfx)

    metrics = LatexInContextMetrics(text_normalizer=normalizer)
    result = metrics.compute_all(predictions=predictions, references=labels)
    return result


def create_diagrams(history, cfg):
    loss = [entry["eval_loss"] for entry in history]
    ortho_wer = [entry["eval_ortho_wer"] for entry in history]
    ortho_cer = [entry["eval_ortho_cer"] for entry in history]
    wer = [entry["eval_wer"] for entry in history]
    cer = [entry["eval_cer"] for entry in history]

    create_diagram("Loss", loss, cfg)


def create_diagram(points, name, path):
    plt.plot(points)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(name)
    plt.savefig(path)


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


@hydra.main(version_base=None, config_path="../configs", config_name="asr_config")
def main(cfg: DictConfig):
    # Creating loggers
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    logger.info(device)

    logger.info("------- Running Experiment Configuration -------")

    if not cfg.get("model"):
        raise ValueError("Missing 'model' configutation block in your YAML")

    # Instantiating model and processor (Can either be a pretrained model or customly trained model)
    logger.info("------- Model Configurations -------")
    logger.info(f"{cfg.model}\n")

    if not cfg.get("processor"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    logger.info(f"{cfg.processor}\n")

    logger.info("------- Instantiating Model from Configuration -------")

    # Model init
    def model_init(trial):
        model = hydra.utils.instantiate(cfg.model)
        model = model(
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        ).to(device)

        if hasattr(model, "config"):
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            model.config.use_cache = False

        if cfg.get("use_lora", False) and cfg.get("lora_config"):
            config = LoraConfig(**cfg.lora_config)
            model = get_peft_model(model, config).to(device)

            trainable_parameters = model.get_nb_trainable_parameters()
            all_parameters = len(model.parameters())
            percentage = all_parameters / trainable_parameters
            logger.info(
                f"Trainable params: {trainable_parameters} | All params: {all_parameters} | Trainable%: {percentage}"
            )

        return model

    architecture = cfg.architecture
    processor = hydra.utils.instantiate(cfg.processor)
    model = model_init(None)
    normalizer = (
        hydra.utils.instantiate(cfg.normalizer) if cfg.get("normalizer") else None
    )
    latex_normalizer = create_latex_normalizer(normalizer=normalizer)

    # Creating Dataset and Dataloader
    if not cfg.get("dataset"):
        raise ValueError("Missing 'data' configutation block in your YAML")

    # Loading in dataset
    datasets = hydra.utils.instantiate(cfg.dataset)
    train, test = datasets.train, datasets.test
    valid = datasets.get("validation", test)

    # Instantiating preprocessing function an then preprocessing the raw dataset
    # Each sample should be in the following format: {input_features/input_values, labels, input_lengths}

    normalize_during_preprocessing = cfg.get("normalize_during_preprocessing", False)
    preprocess_fn = hydra.utils.instantiate(
        cfg.preprocess,
        processor=processor,
        architecture=architecture,
        normalizer=normalizer if normalize_during_preprocessing else None,
    )
    train = preprocess_fn(train)
    valid = preprocess_fn(valid)
    test = preprocess_fn(test)

    # Creating metrics
    compute_metrics = create_metric(processor=processor, normalizer=latex_normalizer)

    # Creating trainer
    trainer = (
        create_ctc_trainer(
            cfg=cfg,
            model=model_init,
            processor=processor,
            train=train,
            valid=valid,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorCTCWithPadding(processor=processor),
            timestamp=timestamp,
        )
        if architecture == "ctc"
        else create_seq2seq_trainer(
            cfg=cfg,
            model=model_init,
            processor=processor,
            train=train,
            valid=valid,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
            timestamp=timestamp,
        )
    )

    # Calculating previous WER scores
    pre_metrics = inference(model, processor, latex_normalizer, test, architecture)
    logger.info(
        f"Previous Results - {' - '.join(f'{metric}: {pre_metrics[metric]:.2f}' for metric in pre_metrics)}"
    )

    # Deleting pre-evaluation model and clearing cache
    del model
    torch.cuda.empty_cache()

    # Execute hyperparameter search
    if cfg.get("use_hyperparameter_search", False):
        n_trials = cfg.get("n_trials", 10)
        logger.info(f"Starting Optuna search with {n_trials} trials...")

        best_run = trainer.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=compute_objective,
            direction="minimize",
            backend="optuna",
            n_trials=n_trials,
        )

        logger.info("------- Best Hyperparameters Found -------")
        logger.info(best_run)

        # Re-train with the best hyperparameters
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

    # Training and logging metrics
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    create_diagrams(trainer.state.log_history, cfg)

    # Evaluate using the validation dataset
    valid_metrics = trainer.evaluate()
    trainer.log_metrics("eval", valid_metrics)
    trainer.save_metrics("eval", valid_metrics)

    # Saving model
    model_directory = cfg.model_directory
    trainer.save_model(os.path.join(model_directory, timestamp))

    # Evaluating finetuned model on test dataset
    logger.info("------- Evaluating Best Model on Test Dataset -------")
    post_metrics = inference(
        trainer.model, processor, latex_normalizer, test, architecture
    )
    logger.info(
        f"Previous Results - {' - '.join(f'{metric}: {post_metrics[metric]:.2f}' for metric in post_metrics)}"
    )


if __name__ == "__main__":
    main()
