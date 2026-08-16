from data.normalizer import contains_equation, has_valid_equation
from transformers import AutoTokenizer


def create_messages(text, label=None):
    messages = [
        {
            "role": "system",
            "content": "You are a specialized translator that outputs LaTeX code based on mathematical sentences.",
        },
        {
            "role": "user",
            "content": f"Convert this mathematical sentence to LaTeX: {text}",
        },
    ]

    if label:
        messages.append({"role": "assistant", "content": f"{label}"})

    return messages


def preprocess_speech2latex(dataset, tokenizer, normalizer):
    dataset = dataset.remove_columns(["audio_path"])
    dataset = dataset.filter(lambda sample: sample["language"] == "eng", num_proc=10)
    dataset = dataset.filter(
        lambda sample: has_valid_equation(sample["sentence"]), num_proc=10
    )
    dataset = dataset.filter(
        lambda sample: contains_equation(sample["sentence"]), num_proc=10
    )

    def preprocess(batch):
        text = batch["whisper_text"]
        label = batch["sentence"]

        if normalizer:
            text = normalizer(text)
            label = normalizer(label)

        messages = create_messages(text=text)
        full_messages = create_messages(text=text, label=label)

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
