from datasets.features import Audio
from data.normalizer import contains_equation, has_valid_equation


def preprocess(dataset, processor, architecture, normalizer):
    input_key = "input_values" if architecture == "ctc" else "input_features"
    target_sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    def preprocess(batch):
        # Extract raw audio arrays from the nested 'audio' dictionary column
        audio = batch["audio"]
        array = audio["array"]

        text = batch["text"]
        if normalizer:
            text = normalizer(text)

        # Process the audio to generate 'input_features' or 'input_values'
        batch = processor(
            audio=audio["array"],
            sampling_rate=target_sampling_rate,
            text=text,
            return_tensors="pt",
        )

        # Remove batch dimension from the input_features and labels
        batch[input_key] = batch[input_key].squeeze(dim=0)
        batch["labels"] = batch["labels"].squeeze(dim=0)

        # Hugging Face models expect text targets to be named 'labels'
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        return batch

    # Map the preprocessing function across the entire dataset in batches
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.with_format(type="torch")
    return dataset


def preprocess_speech2latex(dataset, processor, architecture, normalizer=None):
    target_sampling_rate = processor.feature_extractor.sampling_rate

    # 1. Cast audio column for auto-decoding
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    # 2. Combine all filtering into a SINGLE pass for high efficiency
    def combined_filter(sample):
        # Language check
        if sample["language"] != "eng":
            return False

        # Equation quality checks
        text = sample["sentence"]
        if not (contains_equation(text) and has_valid_equation(text)):
            return False

        # Single-channel check (HF datasets loads audio as 1D numpy array shape (N,) for mono)
        # Multi-channel arrays would have ndim == 2
        audio_data = sample["audio_path"].get_all_samples().data
        if audio_data.ndim != 1:
            return False

        return True

    dataset = dataset.filter(combined_filter, num_proc=10)

    # 3. Corrected and vectorized batched mapping
    def preprocess(batch):
        # Extract audio arrays directly from Hugging Face's pre-decoded structures
        audio_list = [sample.get_all_samples().data.squeeze(dim=0) for sample in batch["audio_path"]]
        texts = batch["sentence"]

        if normalizer:
            texts = [normalizer(text) for text in texts]

        # Run HF Processor
        model_inputs = processor.feature_extractor(
            audio_list,
            sampling_rate=target_sampling_rate,
            return_tensors=None,  # Return raw python lists/numpy arrays for variable length dataset saving
        )

        # Tokenize labels without padding
        labels = processor.tokenizer(texts, return_tensors=None).input_ids
        model_inputs["labels"] = labels

        model_inputs["input_length"] = [
            len(arr) / target_sampling_rate for arr in audio_list
        ]

        return model_inputs

    # Map with multiprocessing support
    dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=64,  # Adjust based on system RAM
    )

    dataset = dataset.with_format(type="torch")
    return dataset