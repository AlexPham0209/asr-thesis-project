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


def dedupe_by_sentence_id(dataset, prefer_human: bool = True):
    sentence_ids = dataset["sentence_id"]
    is_tts = dataset["is_tts"] if prefer_human else [0] * len(sentence_ids)

    chosen: dict = {}  # sentence_id -> (is_tts, row index)
    for idx, (sid, tts) in enumerate(zip(sentence_ids, is_tts)):
        current = chosen.get(sid)
        if current is None or (tts < current[0]):
            chosen[sid] = (tts, idx)

    keep = sorted(idx for _, idx in chosen.values())
    return dataset.select(keep)
