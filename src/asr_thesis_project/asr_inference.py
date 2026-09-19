import json
import os
import sys
import logging
import warnings
from datetime import datetime
import functools
import gc

import torch
import torchaudio
import datasets
import hydra
from omegaconf import DictConfig

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from asr_thesis_project.data.filters import combined_filter
from asr_thesis_project.utils.asr import load_whisper, run_asr_batch
from asr_thesis_project.utils.logger import initialize_loggers
from asr_thesis_project.utils.latex_metrics import LatexInContextMetrics

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(
    version_base=None, config_path="../configs", config_name="asr_inference_config"
)
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    logger.info(f"Using primary device: {device}")
    logger.info("------- Initializing Inference Pipeline -------")

    # 1. Load Dataset & Filter
    dataset_name = cfg.get("dataset_name", "marsianin500/Speech2Latex")
    dataset_split = cfg.get("dataset_split", "sentences_test")
    dataset = datasets.load_dataset(dataset_name, name="default", split=dataset_split)

    logger.info("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=cfg.get("num_proc", 10))

    max_samples = cfg.get("max_eval_samples", len(dataset))
    if max_samples:
        logger.info(f"Subsampling to {max_samples} samples.")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    batch_size = cfg.get("batch_size", 8)

    # 2. Stage 1: ASR Processing
    asr_model_id = cfg.get("asr_model_id", "openai/whisper-small")
    logger.info(f"Loading ASR model: {asr_model_id}")

    asr_model, asr_processor, target_sampling_rate = load_whisper(asr_model_id)

    asr_fn = functools.partial(
        run_asr_batch,
        asr_model=asr_model,
        asr_processor=asr_processor,
        target_sampling_rate=target_sampling_rate,
    )

    # Drop original dataset columns during map to avoid Arrow serialization issues
    dataset = dataset.map(
        asr_fn, batched=True, batch_size=batch_size, remove_columns=dataset.column_names
    )

    results_directory = cfg.get("results_directory", "results")
    os.makedirs(results_directory, exist_ok=True)

    # Persist predictions before metrics so they can be inspected / re-scored.
    with open(os.path.join(results_directory, f"predictions_{timestamp}.jsonl"), "w") as f:
        for pred, ref in zip(dataset["raw_asr_predictions"], dataset["references"]):
            f.write(json.dumps({"prediction": pred, "reference": ref}) + "\n")

    # 4. Compute Metrics
    logger.info("Computing Metrics...")
    metrics = LatexInContextMetrics()
    results = {
        "asr_model_id": asr_model_id,
        "n_samples": len(dataset),
        "asr": metrics.compute_all(
            predictions=dataset["raw_asr_predictions"], references=dataset["references"]
        ),
    }

    logger.info("------- Final Evaluation Results -------")
    for metric_name, value in results["asr"].items():
        logger.info(f"{metric_name}: {value}")

    with open(os.path.join(results_directory, f"results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
