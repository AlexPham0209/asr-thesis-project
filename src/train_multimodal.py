from builtins import getattr
from datetime import datetime
import logging
import os
import sys
import time

import optuna
from utils.hyperparameter import (
    compute_objective,
    create_hyperparameter_diagrams,
    hp_space,
)
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
from utils.metrics import (
    create_metric,
    create_llm_metric,
    preprocess_logits_for_metrics,
)

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
                **inputs, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id
            )

        # Strip input prompt tokens from output
        generated_ids = generated_ids[:, inputs.input_ids.shape[-1] :]
        pred_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(pred_str)
        labels.append(label_text)

    model.train()
    metrics = LatexInContextMetrics(text_normalizer=normalizer)
    return metrics.compute_all(predictions=predictions, references=labels)


def create_diagram(points, name, path):
    plt.plot(points)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(name)
    plt.savefig(path)



@hydra.main(
    version_base=None, config_path="../configs", config_name="post_correction_config"
)
def main(cfg: DictConfig):
    # Creating loggers
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    logger.info(device)

    logger.info("------- Running Experiment Configuration -------")

    if not cfg.get("model"):
        raise ValueError("Missing 'model' configutation block in your YAML")

    # Instantiating model and processor (Can either be a pretrained model or customly trained model)
    logger.info("------- Model Configurations -------")
    logger.info(f"{cfg.processor}\n")

    if not cfg.get("processor"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    logger.info(f"{cfg.tokenizer}\n")

    logger.info("------- Instantiating Model from Configuration -------")

    # Model init
    def model_init(trial):
        model = hydra.utils.instantiate(cfg.model)
        return model

    model = model_init(None)
    processor = hydra.utils.instantiate(cfg.processor)
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
        normalizer=normalizer if normalize_during_preprocessing else None,
    )
    train = preprocess_fn(train)
    test = preprocess_fn(test)

    lora_config = (
        LoraConfig(**OmegaConf.to_container(cfg.lora_config, resolve=True))
        if cfg.get("use_lora", False) and cfg.get("lora_config")
        else None
    )

    # Creating metrics
    compute_metrics = create_llm_metric(
        tokenizer=processor, normalizer=latex_normalizer
    )

    # Model name and directory
    model_name = cfg.get("model_name", "model")
    model_name_timestamp = f"{model_name}_{timestamp}"
    model_directory = os.path.join(cfg.model_directory, model_name)

    # Studies storage folder
    studies_directory = os.path.join("studies", model_name)
    os.makedirs(studies_directory, exist_ok=True)

    training_args = SFTConfig(
        **cfg.training,
        max_length=512,
        dataset_text_field="messages",
        assistant_only_loss=True,
        loss_type="nll",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        output_dir=model_directory,
    )

    # Creating trainer
    trainer = SFTTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        peft_config=lora_config,
        compute_metrics=compute_metrics,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

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
            study_name=f"{model_name}_optuna_study",
            storage=f"sqlite:///{studies_directory}/{model_name}_optuna_trials.db",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            load_if_exists=True,
        )

        logger.info("------- Best Hyperparameters Found -------")
        logger.info(best_run)

        create_hyperparameter_diagrams(
            name=model_name,
            model_directory=model_directory,
            studies_directory=studies_directory,
        )

        # Re-train with the best hyperparameters
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

        trainer.model = model_init(None)

    # Training and logging metrics
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    # Evaluate using the validation dataset
    valid_metrics = trainer.evaluate()
    trainer.log_metrics("eval", valid_metrics)
    trainer.save_metrics("eval", valid_metrics)

    # Saving model
    saved_directory = os.path.join(model_directory, "result")
    os.makedirs(saved_directory, exist_ok=True)
    trainer.save_model(saved_directory)


if __name__ == "__main__":
    main()
