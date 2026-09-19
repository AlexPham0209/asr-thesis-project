"""Fine-tune a decoder-only audio LLM (e.g. Qwen2-Audio) for speech -> LaTeX.

Training uses the plain `Trainer` with teacher-forced next-token loss on the
assistant turn only. Checkpoint selection / early stopping use a teacher-forced
CER (cheap, no generation). After training, the test split is scored with real
`generate()` decoding so the reported CER is comparable to the inference scripts.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta

import accelerate
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from asr_thesis_project.data.data_collator import DataCollatorAudioLMWithPadding
from asr_thesis_project.data.normalizer import create_latex_normalizer
from asr_thesis_project.utils.hyperparameter import (
    compute_objective,
    create_hyperparameter_diagrams,
    hp_space,
)
from asr_thesis_project.utils.latex_metrics import LatexInContextMetrics
from asr_thesis_project.utils.logger import CustomLoggingCallback, initialize_loggers
from asr_thesis_project.utils.metrics import create_llm_metric, preprocess_logits_for_metrics

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("finetuning")
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")


def bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)


@torch.no_grad()
def evaluate(
    model,
    processor,
    dataset,
    sampling_rate: int,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    normalizer=None,
):
    """Real decoding on prompt-only inputs; returns (metrics, predictions, references)."""
    model.eval()
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"  # batched generate needs left padding

    predictions, references = [], []
    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : start + batch_size]
        audios = [np.asarray(a, dtype=np.float32) for a in rows["audio"]]

        inputs = processor(
            text=rows["prompt_text"],
            audio=audios,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if bf16_available() else torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = generated[:, inputs["input_ids"].shape[1] :]
        predictions.extend(
            t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
        references.extend(rows["label"])

    tokenizer.padding_side = "right"
    model.train()

    metrics = LatexInContextMetrics(text_normalizer=normalizer)
    return metrics.compute_all(predictions=predictions, references=references), predictions, references


@hydra.main(version_base=None, config_path="../configs", config_name="multimodal_config")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)
    accelerate.PartialState(timeout=timedelta(seconds=cfg.get("ddp_timeout", 3 * 3600)))

    logger.info(f"Using device: {device}")
    for key in ("model", "processor", "dataset", "preprocess"):
        if not cfg.get(key):
            raise ValueError(f"Missing '{key}' configuration block in your YAML")
    logger.info(f"model: {cfg.model}\nprocessor: {cfg.processor}\n")

    processor = hydra.utils.instantiate(cfg.processor)
    sampling_rate = processor.feature_extractor.sampling_rate
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    use_lora = cfg.get("use_lora", False) and cfg.get("lora_config")

    def model_init(trial=None):
        model = hydra.utils.instantiate(cfg.model, token=HF_TOKEN)
        model.config.use_cache = False  # incompatible with gradient checkpointing

        if use_lora:
            lora_config = LoraConfig(**OmegaConf.to_container(cfg.lora_config, resolve=True))
            model = get_peft_model(model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            logger.info(
                f"Trainable params: {trainable} | All params: {total} | Trainable%: {trainable / total:.4%}"
            )
        elif next(model.parameters()).dtype != torch.float32:
            # AMP needs fp32 master weights for anything that receives gradients.
            raise ValueError(
                "Full fine-tuning requires model.dtype=float32 in the model config "
                "(or set use_lora=true so only fp32 adapters are trained)."
            )
        return model

    normalizer = hydra.utils.instantiate(cfg.normalizer) if cfg.get("normalizer") else None
    latex_normalizer = create_latex_normalizer(normalizer=normalizer)

    datasets = hydra.utils.instantiate(cfg.dataset)
    train, test = datasets.train, datasets.test

    # Validation is carved out of train; the test split is only touched by the
    # final generation-based evaluation.
    if "validation" in datasets:
        valid = datasets.validation
    else:
        split = train.train_test_split(
            test_size=cfg.get("validation_fraction", 0.05), seed=cfg.get("seed", 42)
        )
        train, valid = split["train"], split["test"]

    max_eval = cfg.get("max_eval_samples")
    if max_eval:
        valid = valid.select(range(min(max_eval, len(valid))))
        test = test.select(range(min(max_eval, len(test))))

    preprocess_fn = hydra.utils.instantiate(
        cfg.preprocess,
        processor=processor,
        normalizer=normalizer if cfg.get("normalize_during_preprocessing", False) else None,
        num_proc=cfg.get("num_proc", 10),
    )
    with accelerate.PartialState().main_process_first():
        train = preprocess_fn(train)
        valid = preprocess_fn(valid)
        test = preprocess_fn(test)
    logger.info(f"train={len(train)} valid={len(valid)} test={len(test)}")

    data_collator = DataCollatorAudioLMWithPadding(
        processor=processor, sampling_rate=sampling_rate
    )
    compute_metrics = create_llm_metric(tokenizer=processor.tokenizer, normalizer=latex_normalizer)

    # Training code
    model_name = cfg.get("model_name", "model")
    model_directory_name = (
        f"{model_name}_{timestamp}" if cfg.get("use_timestamp", False) else model_name
    )
    model_directory = os.path.join(cfg.model_directory, model_directory_name)
    studies_directory = os.path.join("studies", model_name)
    os.makedirs(studies_directory, exist_ok=True)

    def build_trainer():
        args = TrainingArguments(
            **cfg.training,
            output_dir=model_directory,
            bf16=bf16_available(),
            fp16=torch.cuda.is_available() and not bf16_available(),
            remove_unused_columns=False,
            label_names=["labels"],
        )
        return Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train,
            eval_dataset=valid,
            data_collator=data_collator,
            processing_class=processor,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[
                CustomLoggingCallback(logger),
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.get("early_stopping_patience", 5)
                ),
            ],
        )

    trainer = build_trainer()

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
            logger.info(f"------- Best Hyperparameters Found -------\n{best_run}")
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
            trainer = build_trainer()  

    train_results = trainer.train(
        resume_from_checkpoint=cfg.get("resume_from_checkpoint", False)
    )
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    if trainer.is_world_process_zero():
        with open(os.path.join(model_directory, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)

    # Teacher-forced validation metrics (same quantity used for checkpoint selection)
    valid_metrics = trainer.evaluate()
    trainer.log_metrics("eval", valid_metrics)
    trainer.save_metrics("eval", valid_metrics)

    saved_directory = os.path.join(model_directory, "result")
    trainer.save_model(saved_directory) 

    if trainer.is_world_process_zero():
        logger.info("Running generation-based evaluation on the test split...")
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.config.use_cache = True
        test_metrics, predictions, references = evaluate(
            model,
            processor,
            test,
            sampling_rate=sampling_rate,
            batch_size=cfg.get("generation_batch_size", 4),
            max_new_tokens=cfg.get("max_new_tokens", 256),
            normalizer=latex_normalizer,
        )

        logger.info("------- Test (generation) results -------")
        for k, v in test_metrics.items():
            logger.info(f"test/{k}: {v}")

        results_directory = os.path.join(cfg.get("results_directory", "results"), model_directory_name)
        os.makedirs(results_directory, exist_ok=True)
        with open(os.path.join(results_directory, "test_results.json"), "w") as f:
            json.dump({"n_samples": len(references), **test_metrics}, f, indent=4)
        with open(os.path.join(results_directory, "test_predictions.jsonl"), "w") as f:
            for pred, ref in zip(predictions, references):
                f.write(json.dumps({"prediction": pred, "reference": ref}) + "\n")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
