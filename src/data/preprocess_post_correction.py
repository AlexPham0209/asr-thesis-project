from data.normalizer import contains_equation, has_valid_equation


def create_messages(text, label=None):
    messages = [
        {"role": "system", "content": "You are a specialized translator that outputs LaTeX code based on mathematical sentences."},
        {"role": "user", "content": f"Convert this mathematical sentence to LaTeX: {text}"},
    ]

    if label:
        messages.append(
            {"role": "assistant", "content": f"{label}"}
        )

    return messages


def preprocess_speech2latex(dataset, tokenizer, normalizer):
    dataset = dataset.remove_columns(['audio'])
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
        batch["text"] = tokenizer.apply_chat_template(full_messages, tokenize=False)
        batch["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return batch

    # Map the preprocessing function across the entire dataset in batches
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.with_format(type="torch")
    return dataset
