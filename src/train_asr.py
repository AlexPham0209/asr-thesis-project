from builtins import getattr
from datetime import datetime
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    EarlyStoppingCallback,
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
from utils.logger import CustomLoggingCallback, initialize_loggers
from utils.metrics import create_metric

from data.data_collator import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)

import logging
from transformers.utils import logging as hf_logging
from peft import get_peft_model, LoraConfig
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_seq2seq_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator, model_directory
):
    training_args = Seq2SeqTrainingArguments(
        **cfg.training,
        output_dir=model_directory,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model_init=model,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[
            CustomLoggingCallback(logger),
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.0
            ),
        ],
    )

    return trainer


def create_ctc_trainer(
    cfg,
    model,
    processor,
    train,
    valid,
    compute_metrics,
    data_collator,
    model_directory,
):
    training_args = TrainingArguments(**cfg.training, output_dir=model_directory)

    trainer = Trainer(
        model_init=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomLoggingCallback(logger),
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.0
            ),
        ],
    )

    return trainer


def inference(model, processor, normalizer, dataset, architecture):
    # Metrics
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    predictions = []
    labels = []
    rtfxs = []

    for sample in dataset:
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


def create_diagram(points, name, path):
    plt.plot(points)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(name)
    plt.savefig(path)


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
        )

        if hasattr(model, "config"):
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            model.config.use_cache = False

        if cfg.get("use_lora", False) and cfg.get("lora_config"):
            config = LoraConfig(**cfg.lora_config)
            model = get_peft_model(model, config)

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

    # Model name and directory
    model_name = cfg.get("model_name", "model")
    model_name_timestamp = f"{model_name}_{timestamp}"
    model_directory = os.path.join(cfg.model_directory, model_name)

    # Studies storage folder
    studies_directory = os.path.join("studies", model_name)
    os.makedirs(studies_directory, exist_ok=True)

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
            model_directory=model_directory,
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
            model_directory=model_directory,
        )
    )

    # # Deleting pre-evaluation model and clearing cache
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

        if trainer.is_world_process_zero() and best_run is not None:
            logger.info("------- Best Hyperparameters Found -------")
            logger.info(best_run)

        # Synchronize to ensure Rank 0 is done drawing diagrams
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            # FIX: Broadcast the best_run object from Rank 0 to all other ranks
            best_run_list = [best_run] if trainer.is_world_process_zero() else [None]
            torch.distributed.broadcast_object_list(best_run_list, src=0)
            best_run = best_run_list[0]

        # Now ALL ranks will evaluate this as True and cleanly re-instantiate the model
        if best_run is not None:
            # Apply best params to args
            for k, v in best_run.hyperparameters.items():
                OmegaConf.update(cfg.training, k, v, merge=True)

            # Safest DDP approach: Re-instantiate the trainer for the final run
            trainer = (
                create_ctc_trainer(
                    cfg=cfg,
                    model=model_init,
                    processor=processor,
                    train=train,
                    valid=valid,
                    compute_metrics=compute_metrics,
                    data_collator=DataCollatorCTCWithPadding(processor=processor),
                    model_directory=model_directory,
                )
                if architecture == "ctc"
                else create_seq2seq_trainer(
                    cfg=cfg,
                    model=model_init,
                    processor=processor,
                    train=train,
                    valid=valid,
                    compute_metrics=compute_metrics,
                    data_collator=DataCollatorSpeechSeq2SeqWithPadding(
                        processor=processor
                    ),
                    model_directory=model_directory,
                )
            )

    # Training model with best run hyperparameters
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    # Evaluate using the validation dataset
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        valid_metrics = trainer.evaluate()

    trainer.log_metrics("eval", valid_metrics)
    trainer.save_metrics("eval", valid_metrics)

    # Saving model
    saved_directory = os.path.join(model_directory, "result")
    os.makedirs(saved_directory, exist_ok=True)
    trainer.save_model(saved_directory)


if __name__ == "__main__":
    main()
