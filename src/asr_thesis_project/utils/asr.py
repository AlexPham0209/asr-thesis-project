"""Whisper ASR stage shared by the RAG inference scripts."""

import gc
import logging
import os

from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from asr_thesis_project.utils.audio import decode_batch_audio

logger = logging.getLogger("inference")


def load_whisper(model_id: str):
    adapter_config = os.path.join(model_id, "adapter_config.json")
    if os.path.isdir(model_id) and os.path.exists(adapter_config):
        base_model_id = PeftConfig.from_pretrained(model_id).base_model_name_or_path
        logger.info(f"Loading ASR base model {base_model_id} + LoRA adapter {model_id}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model_id, device_map="auto")
        model = PeftModel.from_pretrained(model, model_id).merge_and_unload()
        processor_id = model_id if os.path.exists(os.path.join(model_id, "preprocessor_config.json")) else base_model_id
    else:
        logger.info(f"Loading ASR model: {model_id}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, device_map="auto")
        processor_id = model_id

    processor = AutoProcessor.from_pretrained(processor_id)
    return model, processor, processor.feature_extractor.sampling_rate


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_asr_batch(batch, asr_model, asr_processor, target_sampling_rate):
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
