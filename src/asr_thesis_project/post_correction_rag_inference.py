import functools
import json
import logging
import os
import warnings
from datetime import datetime

import chromadb
from dotenv import load_dotenv
import datasets
import hydra
from omegaconf import DictConfig
import torch

from asr_thesis_project.data.filters import combined_filter
from asr_thesis_project.models.post_correction_rag import PostCorrectionRAG
from asr_thesis_project.utils.asr import load_whisper, release_cuda, run_asr_batch
from asr_thesis_project.utils.latex_metrics import LatexInContextMetrics
from asr_thesis_project.utils.logger import initialize_loggers

load_dotenv()  # GEMINI_API_KEY / HF_TOKEN for ${oc.env:...} in the configs

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"


def run_rag_batch(batch, rag: PostCorrectionRAG, top_n: int):
    """Stage 2: raw ASR predictions -> RAG LaTeX post-correction"""
    predictions = rag.inference(inputs=batch["raw_asr_predictions"], top_n=top_n)
    return {"predictions": predictions}


def check_collection_matches_embedder(collection, cfg):
    """Refuse to query an index built by a different embedder."""
    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{collection.name}' at {cfg.db_path} is empty. Build it first with "
            "`python -m asr_thesis_project.data.generate_text_rag_dataset`."
        )
    meta = collection.metadata or {}
    built_with = meta.get("embedder")
    expected = cfg.embedding._target_
    if built_with and built_with != expected:
        raise RuntimeError(
            f"Collection '{collection.name}' was built with {built_with}, "
            f"but the query embedder is {expected}."
        )

    # Same class can wrap different checkpoints of the same dimension
    # (bge vs e5, both 768-d), so the model name has to match too.
    built_model = meta.get("embedder_model")
    expected_model = cfg.embedding.get("model_name")
    if built_model and expected_model and built_model != expected_model:
        raise RuntimeError(
            f"Collection '{collection.name}' was built with model {built_model!r}, "
            f"but the query embedder uses {expected_model!r}."
        )


@hydra.main(
    version_base=None, config_path="../configs", config_name="post_correction_rag_config"
)
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    logger.info(f"Using primary device: {device}")
    logger.info("------- Initializing Post-Correction RAG Inference Pipeline -------")

    # 1. Dataset
    dataset_name = cfg.get("dataset_name", "marsianin500/Speech2Latex")
    dataset_split = cfg.get("dataset_split", "sentences_test")
    logger.info(f"Loading dataset: {dataset_name} ({dataset_split})")
    dataset = datasets.load_dataset(dataset_name, name="default", split=dataset_split)

    logger.info("Filtering dataset...")
    dataset = dataset.filter(combined_filter, num_proc=cfg.get("num_proc", 10))

    max_samples = cfg.get("max_eval_samples")
    if max_samples:
        logger.info(f"Subsampling to {max_samples} samples.")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    batch_size = cfg.get("batch_size", 8)
    results_directory = cfg.get("results_directory", "results")
    os.makedirs(results_directory, exist_ok=True)

    # 2. Stage 1: ASR
    asr_model, asr_processor, target_sampling_rate = load_whisper(
        cfg.get("asr_model_id", "openai/whisper-small")
    )

    logger.info("Executing Stage 1: ASR inference...")
    asr_fn = functools.partial(
        run_asr_batch,
        asr_model=asr_model,
        asr_processor=asr_processor,
        target_sampling_rate=target_sampling_rate,
    )
    dataset = dataset.map(
        asr_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,  # drop the torchcodec audio objects
    )

    # Keep the raw ASR output: it's the baseline every RAG number is compared against.
    with open(os.path.join(results_directory, f"asr_{timestamp}.jsonl"), "w") as f:
        for raw, ref in zip(dataset["raw_asr_predictions"], dataset["references"]):
            f.write(json.dumps({"raw_asr": raw, "reference": ref}) + "\n")

    del asr_model, asr_processor
    release_cuda()

    # 3. Stage 2: vector store + embedder + generator
    client = chromadb.PersistentClient(path=cfg.db_path)
    collection = client.get_collection(name=cfg.collection_name)  # no silent create
    check_collection_matches_embedder(collection, cfg)

    embedding = hydra.utils.instantiate(cfg.embedding)
    generator = hydra.utils.instantiate(cfg.generator)

    rag = PostCorrectionRAG(
        system_prompt=cfg.system_prompt,
        collection=collection,
        generator=generator,
        embedding=embedding,
    )

    top_n = cfg.get("top_n", 3)
    logger.info(f"Executing Stage 2: RAG post-correction (top_n={top_n})...")
    dataset = dataset.map(
        functools.partial(run_rag_batch, rag=rag, top_n=top_n),
        batched=True,
        batch_size=batch_size,
    )

    # Persist predictions before metrics so a metrics crash can't lose the API calls.
    with open(os.path.join(results_directory, f"predictions_{timestamp}.jsonl"), "w") as f:
        for raw, pred, ref in zip(
            dataset["raw_asr_predictions"], dataset["predictions"], dataset["references"]
        ):
            f.write(json.dumps({"raw_asr": raw, "prediction": pred, "reference": ref}) + "\n")

    # 4. Metrics — RAG output *and* the raw ASR baseline, side by side.
    logger.info("Computing metrics...")
    metrics = LatexInContextMetrics()
    results = {
        "top_n": top_n,
        "n_samples": len(dataset),
        "rag": metrics.compute_all(
            predictions=dataset["predictions"], references=dataset["references"]
        ),
        "raw_asr": metrics.compute_all(
            predictions=dataset["raw_asr_predictions"], references=dataset["references"]
        ),
    }

    logger.info("------- Final Evaluation Results -------")
    for stage in ("raw_asr", "rag"):
        for metric_name, value in results[stage].items():
            logger.info(f"{stage}/{metric_name}: {value}")

    with open(os.path.join(results_directory, f"results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
