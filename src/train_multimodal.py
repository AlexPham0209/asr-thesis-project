from builtins import getattr
from datetime import datetime
import logging
import os
import sys
import time
import warnings

import optuna
import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,  # Changed for Multimodal LLMs
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from utils.hyperparameter import (
    compute_objective,
    create_hyperparameter_diagrams,
    hp_space,
)
from utils.latex_metrics import LatexInContextMetrics
from data.normalizer import create_latex_normalizer
from utils.logger import CustomLoggingCallback
from utils.metrics import (
    create_metric,
    create_llm_metric,
    preprocess_logits_for_metrics,
)

# You will need a custom multimodal collator (e.g., that stacks input_ids and audio tensors)
from data.data_collator import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"


def inference(model, processor, normalizer, dataset):
    """Inference loop adapted for Multimodal Audio LLMs."""
    predictions = []
    labels = []

    model.eval()
    for sample in dataset:
        # Assuming dataset contains 'audio' (array, sampling_rate) and 'label'
        audio = sample["audio"]
        label_text = sample["label"]
        prompt = sample.get(
            "prompt", "<|audio_bos|><|AUDIO|><|audio_eos|>\nTranscribe:"
        )

        # Process both audio and text prompt simultaneously
        inputs = processor(
            text=prompt,
            audios=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Strip input prompt tokens from output
        input_length = inputs.input_ids.shape[-1]
        generated_ids = generated_ids[:, input_length:]
        pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
    version_base=None, config_path="../configs", config_name="multimodal_asr_config"
)
def main(cfg: DictConfig):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    logger.info(device)
    logger.info("------- Running Experiment Configuration -------")

    if not cfg.get("model") or not cfg.get("processor"):
        raise ValueError(
            "Missing 'model' or 'processor' configuration block in your YAML"
        )

    logger.info("------- Instantiating Model & Processor -------")

    def model_init(trial):
        model = hydra.utils.instantiate(cfg.model)
        return model

    model = model_init(None)
    processor = hydra.utils.instantiate(cfg.processor)
    normalizer = (
        hydra.utils.instantiate(cfg.normalizer) if cfg.get("normalizer") else None
    )
    latex_normalizer = create_latex_normalizer(normalizer=normalizer)

    logger.info("------- Preparing Dataset -------")
    if not cfg.get("dataset"):
        raise ValueError("Missing 'data' configuration block in your YAML")

    datasets = hydra.utils.instantiate(cfg.dataset)
    train, test = datasets.train, datasets.test
    valid = datasets.get("validation", test)

    # Preprocessing should map inputs to input_ids, attention_mask, labels, and audio tensors
    normalize_during_preprocessing = cfg.get("normalize_during_preprocessing", False)
    preprocess_fn = hydra.utils.instantiate(
        cfg.preprocess,
        processor=processor,
        normalizer=normalizer if normalize_during_preprocessing else None,
    )

    train = preprocess_fn(train)
    test = preprocess_fn(test)

    # Load Custom Multimodal Data Collator
    data_collator = None
    if cfg.get("data_collator"):
        data_collator = hydra.utils.instantiate(cfg.data_collator, processor=processor)

    lora_config = (
        LoraConfig(**OmegaConf.to_container(cfg.lora_config, resolve=True))
        if cfg.get("use_lora", False) and cfg.get("lora_config")
        else None
    )

    compute_metrics = create_metric(processor=processor, normalizer=latex_normalizer)

    model_name = cfg.get("model_name", "multimodal_model")
    model_directory = os.path.join(cfg.model_directory, model_name)
    studies_directory = os.path.join("studies", model_name)
    os.makedirs(studies_directory, exist_ok=True)

    # SFTConfig adapted for Multimodal Audio
    training_args = SFTConfig(
        **cfg.training,
        max_length=cfg.get("max_length", 512),
        packing=False,  # MUST BE FALSE FOR AUDIO TENSORS
        remove_unused_columns=False,  # MUST BE FALSE so audio columns aren't deleted by Trainer
        dataset_text_field=None,  # Remove text-field reliance; use preprocess_fn output
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        output_dir=model_directory,
    )

    logger.info("------- Initializing SFTTrainer -------")
    trainer = SFTTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,  # Inject multimodal collator
        peft_config=lora_config,
        compute_metrics=compute_metrics,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Hyperparameter search via Optuna
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

        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)
        trainer.model = model_init(None)

    # Execute Training
    logger.info("------- Starting Training -------")
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    # Execute Evaluation
    logger.info("------- Starting Evaluation -------")
    valid_metrics = trainer.evaluate()
    trainer.log_metrics("eval", valid_metrics)
    trainer.save_metrics("eval", valid_metrics)

    # Save finalized model
    saved_directory = os.path.join(model_directory, "result")
    os.makedirs(saved_directory, exist_ok=True)
    trainer.save_model(saved_directory)


if __name__ == "__main__":
    main()
