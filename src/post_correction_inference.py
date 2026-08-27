import os
import sys
import logging
import warnings
from datetime import datetime
import functools

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
from transformers.utils import logging as hf_logging
from peft import PeftModel

from utils.logger import initialize_loggers
from utils.latex_metrics import LatexInContextMetrics

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"

def combined_filter(sample):
    """Filters dataset for English language and single-channel audio."""
    if sample["language"] != "eng":
        return False

    audio_data = sample["audio_path"].get_all_samples().data
    if not (audio_data.ndim == 2 and audio_data.shape[0] == 1):
        return False

    return True

def evaluate_batch(batch, asr_model, asr_processor, llm_model, llm_tokenizer, system_prompt, target_sampling_rate, eval_device):
    """Batched mapping function for executing ASR followed by LLM post-correction."""
    audios = []
    
    for audio in batch["audio_path"]:
        samples = audio.get_all_samples()
        audio_tensor = samples.data.squeeze(dim=0)
        
        if samples.sample_rate != target_sampling_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=samples.sample_rate, 
                new_freq=target_sampling_rate
            )
        audios.append(audio_tensor.numpy())

    references = batch["whisper_text"]

    # 1. --- ASR Inference ---
    inputs = asr_processor(
        audio=audios,
        sampling_rate=target_sampling_rate,
        return_tensors="pt"
    ).to(eval_device)

    with torch.no_grad():
        generated_ids = asr_model.generate(inputs["input_features"])

    transcriptions = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)

    # 2. --- LLM Post-Correction ---
    messages_batch = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": t}
        ]
        for t in transcriptions
    ]
    
    prompts = [
        llm_tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) 
        for m in messages_batch
    ]

    llm_inputs = llm_tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(eval_device)

    with torch.no_grad():
        llm_outputs = llm_model.generate(
            **llm_inputs,
            max_new_tokens=256,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
            temperature=0.2, 
            do_sample=True
        )

    corrected_transcriptions = []
    for i, output in enumerate(llm_outputs):
        input_len = llm_inputs.input_ids[i].shape[-1]
        generated_tokens = output[input_len:]
        decoded_text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        corrected_transcriptions.append(decoded_text)

    batch["raw_asr_predictions"] = transcriptions         
    batch["predictions"] = corrected_transcriptions       
    batch["references"] = references
    
    return batch


@hydra.main(version_base=None, config_path="../configs", config_name="inference_config")
def main(cfg: DictConfig):
    # Setup loggers
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    initialize_loggers(cfg=cfg, timestamp=timestamp)

    logger.info(f"Using device: {device}")
    logger.info("------- Initializing Inference Pipeline -------")

    # 1. Load ASR Setup
    asr_model_id = cfg.get("asr_model_id", "openai/whisper-small")
    logger.info(f"Loading ASR model: {asr_model_id}")
    
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_model_id, device_map="auto")
    asr_processor = AutoProcessor.from_pretrained(asr_model_id)
    target_sampling_rate = asr_processor.feature_extractor.sampling_rate

    # 2. Load LLM Setup
    llm_base_model = cfg.get("llm_base_model", "meta-llama/Llama-3-8b-Instruct")
    llm_peft_path = cfg.get("llm_peft_path", None) # Path to LoRA adapters saved during training
    system_prompt = cfg.get(
        "system_prompt", 
        "You are an expert transcription editor. Correct the following ASR output for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text."
    )

    logger.info(f"Loading LLM model: {llm_base_model}")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_base_model)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    if llm_peft_path and os.path.exists(llm_peft_path):
        logger.info(f"Applying LoRA weights from: {llm_peft_path}")
        llm_model = PeftModel.from_pretrained(llm_model, llm_peft_path)
    else:
        logger.warning("No valid LoRA path provided or found. Running with base LLM only.")

    llm_model.eval()

    # 3. Load & Filter Dataset
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
    
    # 4. Map the Pipeline (using functools to cleanly pass models)
    logger.info("Executing batched ASR + LLM Post-Correction mapping...")
    
    # Partial function binds the loaded models to the map function natively
    eval_fn = functools.partial(
        evaluate_batch,
        asr_model=asr_model,
        asr_processor=asr_processor,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        system_prompt=system_prompt,
        target_sampling_rate=target_sampling_rate,
        eval_device=device
    )

    # Batch size of 8 or 4 is recommended when loading Whisper + LLM simultaneously on one GPU
    batch_size = cfg.get("batch_size", 8) 
    
    dataset = dataset.map(eval_fn, batched=True, batch_size=batch_size)

    # 5. Compute Metrics
    logger.info("Computing Metrics...")
    metrics = LatexInContextMetrics()

    results = metrics.compute_all(
        predictions=dataset["predictions"], 
        references=dataset["references"]
    )

    logger.info("------- Final Evaluation Results -------")
    for metric_name, value in results.items():
        logger.info(f"{metric_name}: {value}")

if __name__ == "__main__":
    main()