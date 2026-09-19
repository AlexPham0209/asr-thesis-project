import json
import os
import sys
import logging
import warnings
from datetime import datetime
import functools
import gc

from dotenv import load_dotenv
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
from asr_thesis_project.data.preprocess_post_correction import (
    DEFAULT_PROMPT,
    SYSTEM_PROMPT_FILE,
    create_messages,
)
from asr_thesis_project.utils.asr import load_whisper, run_asr_batch
from asr_thesis_project.utils.logger import initialize_loggers
from asr_thesis_project.utils.latex_metrics import LatexInContextMetrics

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")



def run_llm_batch(batch, llm_model, llm_tokenizer, system_prompt):
    """Stage 2: Raw ASR Predictions -> LaTeX Corrected Output"""
    transcriptions = batch["raw_asr_predictions"]

    messages_batch = [create_messages(text=t, system_prompt=system_prompt) for t in transcriptions]

    prompts = [
        llm_tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]

    llm_inputs = llm_tokenizer(
        prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True
    ).to(llm_model.device)

    with torch.no_grad():
        llm_outputs = llm_model.generate(
            **llm_inputs,
            max_new_tokens=512,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
            do_sample=False,
        )

    prompt_length = llm_inputs.input_ids.shape[-1]
    generated_ids = llm_outputs[:, prompt_length:]
    corrected_transcriptions = llm_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    return {"predictions": corrected_transcriptions}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="post_correction_inference_config",
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

    logger.info(f"ASR done on {len(dataset)} samples; e.g. {dataset[0]['raw_asr_predictions']!r}")

    # Free ASR memory before loading LLM
    del asr_model
    del asr_processor
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Stage 2: LLM Post-Correction
    llm_base_model = cfg.get("llm_base_model", "meta-llama/Llama-3-8b-Instruct")
    llm_peft_path = cfg.get("llm_peft_path", None)
    logger.info(f"Loading LLM model: {llm_base_model}")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_peft_path 
        if llm_peft_path and os.path.exists(llm_peft_path) 
        else llm_base_model
    )
    llm_tokenizer.padding_side = "left"
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if llm_peft_path and os.path.exists(llm_peft_path):
        logger.info(f"Applying LoRA weights from: {llm_peft_path}")
        llm_model = PeftModel.from_pretrained(llm_model, llm_peft_path)
    elif llm_peft_path:
        raise ValueError("Invalid PEFT path")
        
    llm_model.eval()

    logger.info("Executing Stage 2: LLM Post-Correction...")
    system_prompt = cfg.get("system_prompt", DEFAULT_PROMPT)
    saved_prompt_path = os.path.join(llm_peft_path, SYSTEM_PROMPT_FILE) if llm_peft_path else None
    if saved_prompt_path and os.path.exists(saved_prompt_path):
        with open(saved_prompt_path) as f:
            saved_prompt = f.read()
        if saved_prompt.strip() != system_prompt.strip():
            logger.warning(
                "Config system_prompt differs from the one the adapter was trained on; "
                f"using the trained prompt from {saved_prompt_path}."
            )
        system_prompt = saved_prompt
    elif llm_peft_path:
        logger.warning(
            f"No {SYSTEM_PROMPT_FILE} next to the adapter; assuming it was trained with the "
            "config/default prompt. Adapters trained before this file existed used DEFAULT_PROMPT."
        )
    llm_fn = functools.partial(
        run_llm_batch,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        system_prompt=system_prompt
    )

    dataset = dataset.map(llm_fn, batched=True, batch_size=batch_size)

    results_directory = cfg.get("results_directory", "results")
    os.makedirs(results_directory, exist_ok=True)

    # Persist predictions before metrics so they can be inspected / re-scored.
    with open(os.path.join(results_directory, f"predictions_{timestamp}.jsonl"), "w") as f:
        for raw, pred, ref in zip(
            dataset["raw_asr_predictions"], dataset["predictions"], dataset["references"]
        ):
            f.write(json.dumps({"raw_asr": raw, "prediction": pred, "reference": ref}) + "\n")

    # 4. Compute Metrics — corrected output and the raw ASR baseline side by side.
    logger.info("Computing Metrics...")
    metrics = LatexInContextMetrics()
    results = {
        "asr_model_id": asr_model_id,
        "llm_base_model": llm_base_model,
        "llm_peft_path": llm_peft_path,
        "n_samples": len(dataset),
        "post_correction": metrics.compute_all(
            predictions=dataset["predictions"], references=dataset["references"]
        ),
        "raw_asr": metrics.compute_all(
            predictions=dataset["raw_asr_predictions"], references=dataset["references"]
        ),
    }

    logger.info("------- Final Evaluation Results -------")
    for stage in ("raw_asr", "post_correction"):
        for metric_name, value in results[stage].items():
            logger.info(f"{stage}/{metric_name}: {value}")

    with open(os.path.join(results_directory, f"results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
