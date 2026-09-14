"""Preprocessing for decoder-only audio LLMs (Qwen2-Audio and friends).

Each row becomes:
  audio          float32 waveform at the processor's sampling rate
  prompt_text    chat-template string up to and including the assistant header
  full_text      prompt_text + label + end-of-turn
  prompt_length  number of tokens in prompt_text *after* audio-token expansion
  label          the LaTeX sentence (kept for generation-based evaluation)

The heavy processor call (mel features + audio-token expansion) is deferred to
DataCollatorAudioLMWithPadding so we don't persist 128x3000 feature maps per row.
"""

import logging

from asr_thesis_project.data.filters import combined_filter
from asr_thesis_project.utils.audio import decode_audio

logger = logging.getLogger("finetuning")

SYSTEM_PROMPT = (
    "You are an expert transcriber of spoken mathematics. Transcribe the audio into "
    "plain English text where every mathematical expression is written as inline "
    "LaTeX enclosed in $...$. Output only the transcription."
)


def create_messages(label: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    if label is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": label}]}
        )
    return messages


def _common_prefix_length(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def preprocess_speech2latex(dataset, processor, normalizer=None, num_proc: int = 10):
    sampling_rate = processor.feature_extractor.sampling_rate

    dataset = dataset.filter(combined_filter, num_proc=num_proc)

    prompt_text = processor.apply_chat_template(
        create_messages(), tokenize=False, add_generation_prompt=True
    )

    def preprocess(sample):
        wav, _ = decode_audio(sample["audio_path"], sampling_rate)
        audio = wav.numpy()

        label = sample["sentence"]
        if normalizer:
            label = normalizer(label)

        full_text = processor.apply_chat_template(
            create_messages(label), tokenize=False, add_generation_prompt=False
        )

        # The number of <|AUDIO|> placeholder tokens depends on the clip length,
        # so the prompt must be tokenized *with* the audio to get its true length.
        prompt_ids = processor(
            text=prompt_text, audio=audio, sampling_rate=sampling_rate
        )["input_ids"][0]
        full_ids = processor(
            text=full_text, audio=audio, sampling_rate=sampling_rate
        )["input_ids"][0]

        prompt_length = len(prompt_ids)
        if list(full_ids[:prompt_length]) != list(prompt_ids):
            # BPE merged across the prompt/answer boundary; mask the shared prefix.
            prompt_length = _common_prefix_length(list(full_ids), list(prompt_ids))
            logger.warning(
                f"Prompt is not a token prefix of the full sequence; masking {prompt_length} tokens."
            )

        return {
            "audio": audio,
            "prompt_text": prompt_text,
            "full_text": full_text,
            "prompt_length": prompt_length,
            "label": label,
        }

    return dataset.map(
        preprocess, remove_columns=dataset.column_names, num_proc=num_proc
    )
