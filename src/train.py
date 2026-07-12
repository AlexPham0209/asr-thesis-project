from datetime import datetime
import logging
import os
import sys
import time

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
from metrics.asr_metric import create_metric

from data.data_collator import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)

import logging
from transformers.utils import logging as hf_logging

logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_seq2seq_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator
):
    training_args = Seq2SeqTrainingArguments(**cfg.training)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    return trainer


def create_ctc_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator
):
    training_args = TrainingArguments(**cfg.training)

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    return trainer


def create_custom_trainer(
    cfg, model, processor, train, valid, compute_metrics, data_collator
):
    pass


def inference(model, processor, dataset, architecture):
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
                predicted_ids = model.generate(input_features)

        end_time = time.perf_counter()

        # Ensure tensor type and add batch dimension for the labels
        label_ids = sample["labels"]

        if not isinstance(label_ids, torch.Tensor):
            label_ids = torch.tensor(label_ids)
        
        label_ids = label_ids.unsqueeze(0)

        # Decoding prediction and labels
        pred_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        audio_duration = sample["input_length"]
        processing_time = end_time - start_time
        rtfx = audio_duration / processing_time

        predictions.extend(pred_str)
        labels.extend(label_str)
        rtfxs.append(rtfx)

    wer_score = wer.compute(predictions=predictions, references=labels)
    cer_score = cer.compute(predictions=predictions, references=labels)
    average_rtfx = torch.tensor(rtfxs).mean(dim=-1)

    return wer_score, cer_score, average_rtfx


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging_directory = cfg.logging_directory
    os.makedirs(logging_directory, exist_ok=True)
    
    # Common log formatter
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    # Creating subfolder for current training run
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = os.path.join(logging_directory, timestamp)
    os.makedirs(run_directory, exist_ok=True)

    # Screen/Console Handler (Attached to root so everything prints to stdout)
    screen_handler = logging.StreamHandler(stream=sys.stdout) 
    screen_handler.setFormatter(file_formatter)

    # Root Logger Setup (Captures everything)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(screen_handler)
    
    root_file_handler = logging.FileHandler(os.path.join(run_directory, "all.log"), mode="w")
    root_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(root_file_handler)

    # Application Logger Setup (Isolates your app's code logs via "finetuning")
    app_logger = logging.getLogger("finetuning")
    app_file_handler = logging.FileHandler(os.path.join(run_directory, "app.log"), mode="w")
    app_file_handler.setFormatter(file_formatter)
    app_logger.addHandler(app_file_handler)

    # Hugging Face Logger Setup (Isolates Hugging Face transformers logs)
    hf_logger_instance = hf_logging.get_logger("transformers")
    hf_file_handler = logging.FileHandler(os.path.join(run_directory, "hf.log"), mode="w")
    hf_file_handler.setFormatter(file_formatter)
    hf_logger_instance.addHandler(hf_file_handler)
    
    hf_logging.set_verbosity_info()
    
    logger.info("------- Running Experiment Configuration -------")
    logger.info(OmegaConf.to_yaml(cfg))

    if not cfg.get("model"):
        raise ValueError("Missing 'model' configutation block in your YAML")

    # Instantiating model and processor (Can either be a pretrained model or customly trained model)
    logger.info("------- Model Configurations -------")
    logger.info(f"{cfg.model}\n")

    if not cfg.get("processor"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    print(f"{cfg.processor}\n")

    logger.info("------- Instantiating Model from Configuration -------")
    architecture = cfg.architecture
    processor = hydra.utils.instantiate(cfg.processor)
    model = hydra.utils.instantiate(cfg.model).to(device)

    # Creating Dataset and Dataloader
    if not cfg.get("dataset"):
        raise ValueError("Missing 'data' configutation block in your YAML")

    # Loading in dataset
    datasets = hydra.utils.instantiate(cfg.dataset)
    train, valid, test = datasets.train, datasets.validation, datasets.test

    # Instantiating preprocessing function an then preprocessing the raw dataset
    # Each sample should be in the following format: {input_features/input_values, labels, input_lengths}
    preprocess_fn = hydra.utils.instantiate(cfg.preprocess, processor=processor, architecture=architecture)
    train = preprocess_fn(train)
    valid = preprocess_fn(valid)
    test = preprocess_fn(test)

    # Creating metrics
    compute_metrics = create_metric(processor=processor)

    # Creating trainer
    trainer = (
        create_ctc_trainer(
            cfg=cfg,
            model=model,
            processor=processor,
            train=train,
            valid=valid,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorCTCWithPadding(processor=processor),
        )
        if architecture == "ctc"
        else create_seq2seq_trainer(
            cfg=cfg,
            model=model,
            processor=processor,
            train=train,
            valid=valid,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        )
    )

    # Calculating previous WER scores
    pre_wer, pre_cer, pre_rtfx = inference(model, processor, test, architecture)

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
    post_wer, post_cer, post_rtfx = inference(model, processor, test, architecture)


if __name__ == "__main__":
    main()