"""Whisper ASR stage shared by the RAG inference scripts."""

import gc
import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from asr_thesis_project.utils.audio import decode_batch_audio

logger = logging.getLogger("inference")


def load_whisper(model_id: str):
    """Returns (model, processor, feature-extractor sampling rate)."""
    logger.info(f"Loading ASR model: {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, processor.feature_extractor.sampling_rate


def release_cuda() -> None:
    """Call after `del`-ing a model to hand its GPU memory back before the next stage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_asr_batch(batch, asr_model, asr_processor, target_sampling_rate):
    """Stage 1 map function: audio -> raw ASR predictions (+ references)."""
    audios = [
        wav.numpy() for wav in decode_batch_audio(batch["audio_path"], target_sampling_rate)
    ]

    inputs = asr_processor(
        audio=audios, sampling_rate=target_sampling_rate, return_tensors="pt"
    ).to(asr_model.device)

    with torch.no_grad():
        generated_ids = asr_model.generate(
            inputs["input_features"], language="english", task="transcribe"
        )

    transcriptions = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {
        "raw_asr_predictions": [t.strip() for t in transcriptions],
        "references": batch["sentence"],
    }
