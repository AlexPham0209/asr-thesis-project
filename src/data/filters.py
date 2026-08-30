def combined_filter(sample):
    # Language check
    if sample["language"] != "eng":
        return False

    # Equation quality checks
    text = sample["whisper_text"]

    audio_data = sample["audio_path"].get_all_samples().data
    if not (audio_data.ndim == 2 and audio_data.shape[0] == 1):
        return False

    return True
