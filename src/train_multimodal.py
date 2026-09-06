import os
import json
import time
import logging
import warnings
import functools
from builtins import getattr
from datetime import datetime

from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import optuna

from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.utils import logging as hf_logging
from peft import get_peft_model, LoraConfig

# Note: trl and SFTTrainer are removed as they are specialized for Text-only RLHF/SFT
# We use standard Seq2SeqTrainer which natively supports predict_with_generate for Audio-to-Text

from utils.hyperparameter import (
    compute_objective,
    create_hyperparameter_diagrams,
    hp_space,
)
from utils.latex_metrics import LatexInContextMetrics
from data.normalizer import create_latex_normalizer
from utils.logger import CustomLoggingCallback, initialize_loggers

# We swap out create_llm_metric for a generic create_metric (usually Word Error Rate / CER for speech)
from utils.metrics import (
    create_metric,
    preprocess_logits_for_metrics,
)

# Crucial for padding variable-length audio features and text labels differently
from data.data_collator import DataCollatorSpeechCausalLMWithPadding, DataCollatorSpeechSeq2SeqWithPadding

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")


def inference(model, processor, normalizer, dataset):
    """Inference loop updated for Speech-to-Text (e.g., QwenAudio)."""
    predictions = []
    labels = []

    model.eval()
    for sample in dataset:
        # Assuming the dataset returns raw audio arrays and target text
        audio = sample["audio"] 
        label_text = sample["label"]

        # Process audio directly into input_features/input_values depending on the model
        inputs = processor(
            audios=audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # For Encoder-Decoder models, we don't need to strip the prompt. 
        # For Decoder-only multimodal models (like QwenAudio), you might still need to split.
        if model.config.is_encoder_decoder:
            pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # Strip prompt tokens if it's a causal LM architecture
            # Adjust index based on how the processor constructs inputs
            generated_ids = generated_ids[:, inputs.input_ids.shape[-1] :]
            pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(pred_str)
        labels.append(label_text)

    model.train()
    metrics = LatexInContextMetrics(text_normalizer=normalizer)
    return metrics.compute_all(predictions=predictions, references=labels)


@hydra.main(
    version_base=None, config_path="../configs", config_name="speech_finetune_config"
)
def main(cfg: DictConfig):
    # Creating loggers
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    logger.info(f"Using device: {device}")
    logger.info("------- Running Experiment Configuration -------")

    if not cfg.get("model"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    if not cfg.get("processor"):
        raise ValueError("Missing 'processor' configuration block in your YAML")

    logger.info("------- Model & Processor Configurations -------")
    logger.info(f"{cfg.model}\n")
    logger.info(f"{cfg.processor}\n")

    # Model init wrapper for Optuna HPO
    def model_init(trial):
        # Hydra will instantiate your model, e.g., AutoModelForCausalLM (QwenAudio) or AutoModelForSpeechSeq2Seq
        model = hydra.utils.instantiate(cfg.model, token=HF_TOKEN)
        
        if cfg.get("use_lora", False) and cfg.get("lora_config"):
            lora_config = OmegaConf.to_container(cfg.lora_config, resolve=True)
            config = LoraConfig(**lora_config)
            model = get_peft_model(model, config)
        
            trainable_parameters, all_parameters = model.get_nb_trainable_parameters()
            percentage = trainable_parameters / all_parameters
            logger.info(
                f"Trainable params: {trainable_parameters} | All params: {all_parameters} | Trainable%: {percentage}"
            )
        return model

    model = model_init(None)
    
    # 1. Swap Tokenizer for Processor (Crucial for Audio models)
    processor = hydra.utils.instantiate(cfg.processor)
    
    normalizer = (
        hydra.utils.instantiate(cfg.normalizer) if cfg.get("normalizer") else None
    )
    latex_normalizer = create_latex_normalizer(normalizer=normalizer)

    if not cfg.get("dataset"):
        raise ValueError("Missing 'dataset' configuration block in your YAML")

    # Loading in dataset
    datasets = hydra.utils.instantiate(cfg.dataset)
    train, test = datasets.train, datasets.test
    valid = datasets.get("validation", test)

    # 2. Update Preprocessing to handle Audio inputs via the processor
    normalize_during_preprocessing = cfg.get("normalize_during_preprocessing", False)
    preprocess_fn = hydra.utils.instantiate(
        cfg.preprocess,
        processor=processor,  # Changed from tokenizer
        normalizer=normalizer if normalize_during_preprocessing else None,
    )
    
    # Assuming preprocess_fn extracts audio features and tokenizes text targets
    train = preprocess_fn(train)
    test = preprocess_fn(test)

    # 4. Metrics setup (using standard create_metric for WER/CER mapping)
    compute_metrics = create_metric(
        processor=processor, 
        normalizer=latex_normalizer
    )

    model_name = cfg.get("model_name", "model")
    model_directory_name = f"{model_name}_{timestamp}" if cfg.get("use_timestamp", False) else model_name
    model_directory = os.path.join(cfg.model_directory, model_directory_name)

    # Studies storage folder
    studies_directory = os.path.join("studies", model_name)
    os.makedirs(studies_directory, exist_ok=True)

    # 5. Use Seq2SeqTrainingArguments for Audio->Text generation capability during eval
    training_args = Seq2SeqTrainingArguments(
        **cfg.training,
        output_dir=model_directory,
        predict_with_generate=True, # Forces the trainer to use model.generate() for evaluation metrics
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # Remove SFT-specific args like dataset_text_field="messages", assistant_only_loss=True
    )

    # 6. Initialize the appropriate Data Collator
    # This handles padding 1D text labels and 2D/3D audio features independently.
    data_collator = DataCollatorSpeechCausalLMWithPadding(processor=processor)

    # 7. Setup the Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        processing_class=processor.tokenizer,  # Trainer requires the tokenizer piece for label decoding internally
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Deleting pre-evaluation model and clearing cache
    del model
    torch.cuda.empty_cache()

    # Optuna Hyperparameter search (Remains effectively the same)
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
            load_if_exists=True,
        )

        if trainer.is_world_process_zero() and best_run is not None:
            logger.info("------- Best Hyperparameters Found -------")
            logger.info(best_run)
            create_hyperparameter_diagrams(
                name=model_name,
                model_directory=model_directory,
                studies_directory=studies_directory,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            best_run_list = [best_run] if trainer.is_world_process_zero() else [None]
            torch.distributed.broadcast_object_list(best_run_list, src=0)
            best_run = best_run_list[0]

        if best_run is not None:
            for k, v in best_run.hyperparameters.items():
                OmegaConf.update(cfg.training, k, v, merge=True)

            # Re-instantiate Trainer with optimal configs
            trainer = Seq2SeqTrainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train,
                eval_dataset=test,
                data_collator=data_collator,
                processing_class=processor.tokenizer,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            )

    # Run Training
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    
    log_history = trainer.state.log_history
    with open(os.path.join(model_directory, "log_history.json"), "w") as f:
        json.dump(log_history, f, indent=4)

    # Evaluate (Automatically triggers predict_with_generate=True for metrics like WER/CER)
    with torch.autocast(device_type=device, dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16):
        valid_metrics = trainer.evaluate()
        
    trainer.log_metrics("eval", valid_metrics)
    trainer.save_metrics("eval", valid_metrics)

    # Save model and processor
    saved_directory = os.path.join(model_directory, "result")
    os.makedirs(saved_directory, exist_ok=True)
    trainer.save_model(saved_directory)
    processor.save_pretrained(saved_directory)


if __name__ == "__main__":
    main()