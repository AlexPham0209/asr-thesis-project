import torchaudio

from data.normalizer import contains_equation, has_valid_equation
from transformers import AutoTokenizer

from data.filters import combined_filter


def create_messages(label):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert transcription editor. Correct the following clip of audio for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio"},
            ],
        },
    ]

    if label:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": label}]}
        )

    return messages


def preprocess_speech2latex(dataset, processor, normalizer):
    # 1. Cast audio column for auto-decoding
    target_sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.filter(combined_filter, num_proc=10)

    def preprocess(batch):
        samples = batch["audio_path"].get_all_samples()
        audio = samples.data.squeeze(dim=0)

        if samples.sample_rate != target_sampling_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=samples.sample_rate, new_freq=target_sampling_rate
            )

        label = batch["sentence"]

        if normalizer:
            label = normalizer(label)

        # Create conversational format for message
        messages = create_messages()
        full_messages = create_messages(label=label)

        # Convert to string via chat template
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        label = processor.apply_chat_template(
            full_messages, add_generation_prompt=True, tokenize=False
        )

        batch["messages"] = full_messages
        batch["text"] = text
        batch["input"] = processor(
            text=text, audios=audio, return_tensors="pt", padding=True
        )
        batch["label"] = label

        return batch

    # Map the preprocessing function across the entire dataset in batches
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset
