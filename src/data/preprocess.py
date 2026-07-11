from datasets.features import Audio


def preprocess_whisper(dataset, processor):
    target_sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    def preprocess(batch):
        # Extract raw audio arrays from the nested 'audio' dictionary column
        audio = batch["audio"]

        # Process the audio to generate 'input_features' or 'input_values'
        batch = processor(
            audio=audio["array"],
            sampling_rate=target_sampling_rate,
            text=batch["text"],
            return_tensors="pt",
        )

        # Remove batch dimension from the input_features and labels
        batch["input_features"] = batch["input_features"].squeeze(dim=0)
        batch["labels"] = batch["labels"].squeeze(dim=0)

        # Hugging Face models expect text targets to be named 'labels'
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        return batch

    # Map the preprocessing function across the entire dataset in batches
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.with_format(type="torch")
    return dataset


def preprocess_wav2vec2(dataset, processor):
    target_sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    def preprocess(batch):
        # Extract raw audio arrays from the nested 'audio' dictionary column
        audio = batch["audio"]

        # Process the audio to generate 'input_features' or 'input_values'
        batch = processor(
            audio=audio["array"],
            sampling_rate=target_sampling_rate,
            text=batch["text"],
            return_tensors="pt",
        )

        # Remove batch dimension from the input_features and labels
        batch["input_values"] = batch["input_values"].squeeze(dim=0)
        batch["labels"] = batch["labels"].squeeze(dim=0)

        # Hugging Face models expect text targets to be named 'labels'
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
        return batch

    # Map the preprocessing function across the entire dataset in batches
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.with_format(type="torch")
    return dataset
