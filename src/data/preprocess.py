from datasets.features import Audio


def preprocess_librispeech(dataset, processor):
    target_sampling_rate = processor.feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    def preprocess(batch):
        # Extract raw audio arrays from the nested 'audio' dictionary column
        audio = batch["audio"]
        audio_arrays = [sample["array"] for sample in audio]

        # Process the audio to generate 'input_features' or 'input_values'
        inputs = processor(
            audio=audio_arrays, 
            sampling_rate=target_sampling_rate, 
            text=batch["text"],
            max_length=10000,
            truncation=True
        )
        
        # Hugging Face models expect text targets to be named 'labels'
        inputs["labels"] = inputs["labels"]
        inputs["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        
        return inputs
    
    # Map the preprocessing function across the entire dataset in batches
    return dataset.map(preprocess, remove_columns=dataset.column_names)
