from builtins import getattr
from datetime import datetime
import logging
import os
import sys
import time
from utils.latex_metrics import LatexInContextMetrics
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
from trl import SFTTrainer, SFTConfig
import functools
import trl

# trl.trainer.sft_trainer._patch_chunked_ce_lm_head = lambda *args, **kwargs: None
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def inference(model, tokenizer, normalizer, dataset):
    """Inference loop specifically for LLM Text-to-Text Post-Correction."""
    predictions = []
    labels = []

    model.eval()
    for sample in dataset:
        input_text = sample["input"] 
        label_text = sample["label"]

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id
            )

        # Strip input prompt tokens from output
        generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
        pred_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(pred_str)
        labels.append(label_text)

    metrics = LatexInContextMetrics(text_normalizer=normalizer)
    return metrics.compute_all(predictions=predictions, references=labels)


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


@hydra.main(version_base=None, config_path="../configs", config_name="post_correction_config")
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

    if not cfg.get("tokenizer"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    logger.info(f"{cfg.tokenizer}\n")

    logger.info("------- Instantiating Model from Configuration -------")

    # Model init
    def model_init(trial):
        model = hydra.utils.instantiate(cfg.model)
        return model

    model = model_init(None)
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    normalizer = (
        hydra.utils.instantiate(cfg.normalizer) if cfg.get("normalizer") else None
    )
    latex_normalizer = create_latex_normalizer(normalizer=normalizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        tokenizer=tokenizer,
        normalizer=normalizer if normalize_during_preprocessing else None,
    )
    train = preprocess_fn(train)
    test = preprocess_fn(test)
    lora_config = LoraConfig(**cfg.lora_config) if cfg.get("use_lora", False) and cfg.get("lora_config") else None

    # Creating metrics
    compute_metrics = create_metric(processor=tokenizer, normalizer=latex_normalizer)
    
    llama_3_training_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% generation %}"
                "{{ message['content'] + '<|eot_id|>' }}"
                "{% endgeneration %}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    tokenizer.chat_template = llama_3_training_template

    training_args = SFTConfig(
        **cfg.training, 
        max_length=512,
        dataset_text_field="messages",
        assistant_only_loss=True,
        loss_type="nll",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    # Creating trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        peft_config=lora_config,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # Calculating previous WER scores
    # pre_metrics = inference(model, tokenizer, latex_normalizer, test)
    # logger.info(
    #     f"Previous Results - {' - '.join(f'{metric}: {pre_metrics[metric]:.2f}' for metric in pre_metrics)}"
    # )

    # Deleting pre-evaluation model and clearing cache
    # del model
    # torch.cuda.empty_cache()

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
        trainer.model, tokenizer, latex_normalizer, test
    )
    logger.info(
        f"Previous Results - {' - '.join(f'{metric}: {post_metrics[metric]:.2f}' for metric in post_metrics)}"
    )


if __name__ == "__main__":
    main()
