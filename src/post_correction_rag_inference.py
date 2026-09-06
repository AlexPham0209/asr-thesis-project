import os
import sys
import logging
import warnings
from datetime import datetime
import functools
import gc

import chromadb
import torch
import torchaudio
import datasets
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from models.post_correction_rag import PostCorrectionRAG
from data.filters import combined_filter
from utils.logger import initialize_loggers
from utils.latex_metrics import LatexInContextMetrics

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"


def run_asr_batch(batch, asr_model, asr_processor, target_sampling_rate):
    """Stage 1: Audio -> Raw ASR Predictions"""
    audios = []

    for audio in batch["audio_path"]:
        samples = audio.get_all_samples()
        audio_tensor = samples.data.squeeze(dim=0)

        if samples.sample_rate != target_sampling_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=samples.sample_rate,
                new_freq=target_sampling_rate,
            )
        audios.append(audio_tensor.numpy())

    # Dynamically match target model device
    inputs = asr_processor(
        audio=audios, sampling_rate=target_sampling_rate, return_tensors="pt"
    ).to(asr_model.device)

    with torch.no_grad():
        generated_ids = asr_model.generate(
            inputs["input_features"], language="english", task="transcribe"
        )

    transcriptions = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)

    return {
        "raw_asr_predictions": transcriptions,
        "references": batch["sentence"],
    }


def run_rag_batch(batch, rag: PostCorrectionRAG):
    """Stage 2: Raw ASR Predictions -> RAG LaTeX Post-Correction"""
    transcriptions = batch["raw_asr_predictions"]
    corrected_transcriptions = rag.inference(inputs=transcriptions)
    return {"predictions": corrected_transcriptions}


@hydra.main(version_base=None, config_path="../configs", config_name="inference_config")
def main(cfg: DictConfig):
    # Setup loggers
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    logger.info(f"Using primary device: {device}")
    logger.info("------- Initializing RAG Inference Pipeline -------")

    # 1. Load & Filter Dataset First
    dataset_name = cfg.get("dataset_name", "marsianin500/Speech2Latex")
    dataset_split = cfg.get("dataset_split", "sentences_test")
    logger.info(f"Loading dataset: {dataset_name} ({dataset_split})")

    dataset = datasets.load_dataset(dataset_name, name="default", split=dataset_split)

    logger.info("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=cfg.get("num_proc", 10))

    max_samples = cfg.get("max_eval_samples", len(dataset))
    if max_samples:
        logger.info(f"Subsampling to {max_samples} samples.")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    batch_size = cfg.get("batch_size", 8)

    # 2. Stage 1: ASR Setup & Execution
    asr_model_id = cfg.get("asr_model_id", "openai/whisper-small")
    logger.info(f"Loading ASR model: {asr_model_id}")

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model_id, device_map="auto"
    )
    asr_processor = AutoProcessor.from_pretrained(asr_model_id)
    target_sampling_rate = asr_processor.feature_extractor.sampling_rate

    logger.info("Executing Stage 1: ASR Inference...")
    asr_fn = functools.partial(
        run_asr_batch,
        asr_model=asr_model,
        asr_processor=asr_processor,
        target_sampling_rate=target_sampling_rate,
    )

    # Drop non-standard audio objects to avoid Arrow serialization errors
    dataset = dataset.map(
        asr_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    # Free ASR memory before initializing LLM & Vector DB
    del asr_model
    del asr_processor
    gc.collect()
    torch.cuda.empty_cache()
    
    # Getting system prompt

    # Getting ChromaDB vector database
    db_path = cfg.get("db_path", "./vector_db")
    collection_name = cfg.get("collection_name", "speech2latex")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    # Creating generator
    generator = hydra.utils.instantiate(cfg.generator)
    system_prompt = cfg.get(
        "system_prompt",
        "You are an expert transcription editor. Correct the following ASR output for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text.",
    )

    logger.info("Initializing RAG module...")
    rag = PostCorrectionRAG(
        system_prompt=system_prompt,
        generator=generator, 
        collection=collection
    )

    logger.info("Executing Stage 2: RAG Post-Correction...")
    rag_fn = functools.partial(run_rag_batch, rag=rag)

    dataset = dataset.map(rag_fn, batched=True, batch_size=batch_size)

    logger.info(dataset["predictions"])
    logger.info(dataset["references"])

    # 4. Compute Metrics
    logger.info("Computing Metrics...")
    metrics = LatexInContextMetrics()

    results = metrics.compute_all(
        predictions=dataset["predictions"], references=dataset["references"]
    )

    logger.info("------- Final Evaluation Results -------")
    for metric_name, value in results.items():
        logger.info(f"{metric_name}: {value}")


if __name__ == "__main__":
    main()
