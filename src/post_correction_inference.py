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

    inputs = asr_processor(
        audio=audios, sampling_rate=target_sampling_rate, return_tensors="pt"
    ).to(asr_model.device)

    with torch.no_grad():
        generated_ids = asr_model.generate(
            inputs["input_features"], language="english", task="transcribe"
        )

    transcriptions = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {"raw_asr_predictions": transcriptions, "references": batch["sentence"]}


def run_llm_batch(batch, llm_model, llm_tokenizer, system_prompt):
    """Stage 2: Raw ASR Predictions -> LaTeX Corrected Output"""
    transcriptions = batch["raw_asr_predictions"]

    messages_batch = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": t}]
        for t in transcriptions
    ]

    prompts = [
        llm_tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]

    llm_inputs = llm_tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(llm_model.device)

    with torch.no_grad():
        llm_outputs = llm_model.generate(
            **llm_inputs,
            max_new_tokens=256,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
            temperature=0.2,
            do_sample=True,
        )

    prompt_length = llm_inputs.input_ids.shape[-1]
    generated_ids = llm_outputs[:, prompt_length:]
    corrected_transcriptions = llm_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    return {"predictions": corrected_transcriptions}


@hydra.main(version_base=None, config_path="../configs", config_name="inference_config")
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

    # Drop original dataset columns during map to avoid Arrow serialization issues
    dataset = dataset.map(
        asr_fn, batched=True, batch_size=batch_size, remove_columns=dataset.column_names
    )

    # Free ASR memory before loading LLM
    del asr_model
    del asr_processor
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Stage 2: LLM Post-Correction
    llm_base_model = cfg.get("llm_base_model", "meta-llama/Llama-3-8b-Instruct")
    llm_peft_path = cfg.get("llm_peft_path", None)
    system_prompt = cfg.get(
        "system_prompt",
        "You are an expert transcription editor. Correct the following ASR output for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text.",
    )

    logger.info(f"Loading LLM model: {llm_base_model}")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_base_model)
    llm_tokenizer.padding_side = "left"
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if llm_peft_path and os.path.exists(llm_peft_path):
        logger.info(f"Applying LoRA weights from: {llm_peft_path}")
        llm_model = PeftModel.from_pretrained(llm_model, llm_peft_path)

    llm_model.eval()

    logger.info("Executing Stage 2: LLM Post-Correction...")
    llm_fn = functools.partial(
        run_llm_batch,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        system_prompt=system_prompt,
    )

    dataset = dataset.map(llm_fn, batched=True, batch_size=batch_size)

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
