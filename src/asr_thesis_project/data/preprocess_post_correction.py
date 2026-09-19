from asr_thesis_project.data.normalizer import contains_equation, has_valid_equation
from transformers import AutoTokenizer

from asr_thesis_project.data.filters import combined_filter

DEFAULT_PROMPT = "You are an expert transcription editor. Correct the following ASR output for grammatical errors, mathematical formatting, and LaTeX terminology. Output ONLY the corrected text."
SYSTEM_PROMPT_FILE = "system_prompt.txt"


def create_messages(text, label=None, system_prompt=DEFAULT_PROMPT):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    if label:
        messages.append({"role": "assistant", "content": f"{label}"})

    return messages


def preprocess_speech2latex(dataset, tokenizer, normalizer, system_prompt=DEFAULT_PROMPT):
    dataset = dataset.filter(combined_filter, num_proc=10)

    def preprocess(batch):
        text = batch["whisper_text"]
        label = batch["sentence"]

        if normalizer:
            text = normalizer(text)
            label = normalizer(label)

        messages = create_messages(text=text, system_prompt=system_prompt)
        full_messages = create_messages(text=text, label=label, system_prompt=system_prompt)

        batch["messages"] = full_messages
        batch["text"] = tokenizer.apply_chat_template(full_messages, tokenize=False)
        batch["input"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch["label"] = label

        return batch

    # Map the preprocessing function across the entire dataset in batches
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset
