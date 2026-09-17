"""Multimodal RAG inference: Gemini receives the query audio + retrieved examples.

query_mode selects how the neighbours are found:
  asr_text  Whisper transcript of the query -> MathBERT -> text index   (cascade, default)
  audio     query waveform -> Whisper encoder -> audio index
"""

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

from asr_thesis_project.utils.audio import decode_batch_audio
from asr_thesis_project.data.filters import combined_filter
from asr_thesis_project.models.multimodal_rag import MultiModalRAG
from asr_thesis_project.utils.asr import load_whisper, release_cuda, run_asr_batch
from asr_thesis_project.utils.latex_metrics import LatexInContextMetrics
from asr_thesis_project.utils.logger import initialize_loggers

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_MODES = ("asr_text", "audio")


def run_rag_batch(batch, rag: MultiModalRAG, audio_sampling_rate: int, top_n: int, query_mode: str):
    audios = decode_batch_audio(batch["audio_path"], audio_sampling_rate)

    if query_mode == "asr_text":
        transcripts = batch["raw_asr_predictions"]
        predictions, examples = rag.inference_with_examples(
            inputs=audios, top_n=top_n, retrieval_inputs=transcripts, hints=transcripts
        )
    else:
        predictions, examples = rag.inference_with_examples(inputs=audios, top_n=top_n)

    # The map drops every input column (audio objects can't be re-serialised), so
    # anything we still need downstream has to be returned explicitly.
    out = {
        "predictions": predictions,
        "retrieved_ids": [[ex.id for ex in exs] for exs in examples],
        "retrieved_targets": [[ex.target for ex in exs] for exs in examples],
        "references": batch["references"] if "references" in batch else batch["sentence"],
    }
    if query_mode == "asr_text":
        out["raw_asr_predictions"] = batch["raw_asr_predictions"]
    return out


def check_collection(collection, cfg, embedding, query_mode: str):
    """Refuse to query an index built by a different embedder / sample rate.

    MathBERT and whisper-small both emit 768-d vectors, so Chroma would happily
    return nonsense neighbours without this.
    """
    if collection.count() == 0:
        raise RuntimeError(f"Collection '{collection.name}' at {cfg.db_path} is empty.")

    meta = collection.metadata or {}
    expected = cfg.embedding._target_
    built_with = meta.get("embedder")
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

    if query_mode == "audio":
        built_sr = meta.get("embedder_sampling_rate")
        if built_sr and int(built_sr) != int(embedding.sampling_rate):
            raise RuntimeError(
                f"Collection was embedded at {built_sr} Hz but query embedder uses "
                f"{embedding.sampling_rate} Hz."
            )

    if not meta.get("has_audio"):
        logger.warning(
            f"Collection '{collection.name}' has no stored example audio; Gemini will only "
            "see the examples as text (rebuild the index with store_audio=true for audio examples)."
        )


@hydra.main(version_base=None, config_path="../configs", config_name="multimodal_rag_config")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    query_mode = cfg.get("query_mode", "asr_text")
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {query_mode!r}")
    audio_sampling_rate = int(cfg.audio_sampling_rate)
    top_n = cfg.get("top_n", 3)
    batch_size = cfg.get("batch_size", 8)

    logger.info(f"Using primary device: {device}")
    logger.info(f"------- Multimodal RAG inference (query_mode={query_mode}, top_n={top_n}) -------")

    results_directory = cfg.get("results_directory", "results")
    os.makedirs(results_directory, exist_ok=True)

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

    # 2. Cascade only: Whisper ASR first. Keep audio_path so stage 2 can re-decode.
    if query_mode == "asr_text":
        asr_model, asr_processor, asr_sr = load_whisper(cfg.asr_model_id)
        logger.info("Stage 1: ASR...")
        dataset = dataset.map(
            functools.partial(
                run_asr_batch,
                asr_model=asr_model,
                asr_processor=asr_processor,
                target_sampling_rate=asr_sr,
            ),
            batched=True,
            batch_size=batch_size,
        )
        del asr_model, asr_processor
        release_cuda()

        # Raw ASR is the baseline every RAG number is compared against.
        with open(os.path.join(results_directory, f"asr_{timestamp}.jsonl"), "w") as f:
            for raw, ref in zip(dataset["raw_asr_predictions"], dataset["references"]):
                f.write(json.dumps({"raw_asr": raw, "reference": ref}) + "\n")

    # 3. Vector store + embedder + generator
    client = chromadb.PersistentClient(path=cfg.db_path)
    collection = client.get_collection(name=cfg.collection_name)

    embedding = hydra.utils.instantiate(cfg.embedding)
    check_collection(collection, cfg, embedding, query_mode)

    generator = hydra.utils.instantiate(cfg.generator)
    if int(getattr(generator, "sample_rate", audio_sampling_rate)) != audio_sampling_rate:
        raise RuntimeError(
            "generator.sample_rate must equal audio_sampling_rate; the query clips are "
            "decoded at audio_sampling_rate and sent to the generator as-is."
        )
    if query_mode == "audio" and int(embedding.sampling_rate) != audio_sampling_rate:
        raise RuntimeError(
            "In audio mode embedding.sampling_rate must equal audio_sampling_rate; the "
            "same decoded tensor is used for retrieval and generation."
        )

    rag = MultiModalRAG(
        system_prompt=cfg.system_prompt,
        generator=generator,
        embedding=embedding,
        collection=collection,
    )

    # 4. Retrieve + generate
    logger.info(f"Stage 2: retrieval + generation on {len(dataset)} samples...")
    dataset = dataset.map(
        functools.partial(
            run_rag_batch,
            rag=rag,
            audio_sampling_rate=audio_sampling_rate,
            top_n=top_n,
            query_mode=query_mode,
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names, 
    )

    # Persist predictions before metrics so a metrics crash can't lose the API calls.
    has_asr = "raw_asr_predictions" in dataset.column_names
    with open(os.path.join(results_directory, f"predictions_{timestamp}.jsonl"), "w") as f:
        for i in range(len(dataset)):
            row = dataset[i]
            record = {
                "prediction": row["predictions"],
                "reference": row["references"],
                "retrieved_ids": row["retrieved_ids"],
                "retrieved_targets": row["retrieved_targets"],
            }
            if has_asr:
                record["raw_asr"] = row["raw_asr_predictions"]
            f.write(json.dumps(record) + "\n")

    # 5. Metrics — RAG output and, in cascade mode, the raw ASR baseline side by side.
    logger.info("Computing metrics...")
    metrics = LatexInContextMetrics()
    results = {
        "query_mode": query_mode,
        "top_n": top_n,
        "n_samples": len(dataset),
        "rag": metrics.compute_all(
            predictions=dataset["predictions"], references=dataset["references"]
        ),
    }
    if has_asr:
        results["raw_asr"] = metrics.compute_all(
            predictions=dataset["raw_asr_predictions"], references=dataset["references"]
        )

    logger.info("------- Final Evaluation Results -------")
    for stage in ("raw_asr", "rag"):
        for metric_name, value in results.get(stage, {}).items():
            logger.info(f"{stage}/{metric_name}: {value}")

    with open(os.path.join(results_directory, f"results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
