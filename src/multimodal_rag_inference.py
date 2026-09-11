import json
import os
import sys
import logging
import warnings
from datetime import datetime
import functools
import gc

import chromadb
from dotenv import load_dotenv
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
from models.multimodal_rag import MultiModalRAG
from utils.logger import initialize_loggers
from utils.latex_metrics import LatexInContextMetrics

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")

def run_rag_batch(batch, rag: MultiModalRAG, target_sampling_rate):
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
            audios.append(audio_tensor)
            
    corrected_transcriptions = rag.inference(inputs=audios)
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

    # Getting ChromaDB vector database
    db_path = cfg.get("db_path", "./vector_db")
    collection_name = cfg.get("collection_name", "speech2latex")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Creating generator
    generator = hydra.utils.instantiate(cfg.generator)
    embedding = hydra.utils.instantiate(cfg.embedding)
    system_prompt = cfg.get(
        "system_prompt",
        "You are an expert transcription editor. Correct the following ASR output for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text.",
    )
    
    logger.info("Initializing RAG module...")
    rag = MultiModalRAG(
        system_prompt=system_prompt, generator=generator, embedding=embedding, collection=collection
    )
    
    rag_fn = functools.partial(
        run_rag_batch, 
        rag=rag,
        target_sampling_rate=getattr(embedding, "sampling_rate", 16000)
    )
    dataset = dataset.map(rag_fn, batched=True, batch_size=batch_size)

    # 4. Compute Metrics
    logger.info("Computing Metrics...")
    metrics = LatexInContextMetrics()

    results = metrics.compute_all(
        predictions=dataset["predictions"], references=dataset["references"]
    )

    logger.info("------- Final Evaluation Results -------")
    for metric_name, value in results.items():
        logger.info(f"{metric_name}: {value}")
    
    
    # Saving metrics
    results_directory = cfg.get("results_directory", "results")
    os.makedirs(results_directory, exist_ok=True)
    with open(os.path.join(results_directory, "results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
