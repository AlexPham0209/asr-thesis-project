import torch
import pytest
import numpy as np
from transformers import AutoProcessor
from datasets import Dataset
from src.data.preprocess import preprocess_wav2vec2, preprocess_whisper


@pytest.fixture(scope="module")
def wav2vec2():
    return AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")


@pytest.fixture(scope="module")
def whisper():
    return AutoProcessor.from_pretrained("openai/whisper-tiny")


@pytest.fixture(scope="module")
def sample_audio():
    # Create a 1-second dummy audio signal at 16000Hz
    sampling_rate = 16000
    duration = 1.0
    audio_array = np.random.randn(int(sampling_rate * duration)).astype(np.float32)
    return audio_array, sampling_rate


def test_whisper_loading(whisper):
    """Test that the processor loads correctly."""
    assert whisper is not None

def test_wav2vec2_loading(wav2vec2):
    """Test that the processor loads correctly."""
    assert wav2vec2 is not None

def test_whisper_processor_audio_output(whisper, sample_audio):
    """Test processing raw audio into model inputs."""
    audio_array, sampling_rate = sample_audio

    # Process the audio
    sample = {
        "audio": {"array": audio_array, "sampling_rate": sampling_rate},
        "text": "HELLO THERE HOW ARE YOU",
    }

    dataset = Dataset.from_list([sample])
    dataset = preprocess_whisper(dataset, whisper)

    result = next(iter(dataset))

    assert list(result.keys()) == ["input_features", "labels", "input_length"], (
        "Incorrect keys"
    )
    assert result["input_features"].shape == (1, 80, 3000)
    assert result["input_length"] == 1.0
    assert (
        whisper.decode(result["labels"], skip_special_tokens=True)
        == "HELLO THERE HOW ARE YOU"
    )


def test_wav2vec2_processor_audio_output(wav2vec2, sample_audio):
    """Test processing raw audio into model inputs."""
    audio_array, sampling_rate = sample_audio

    # Process the audio
    sample = {
        "audio": {"array": audio_array, "sampling_rate": sampling_rate},
        "text": "HELLO THERE HOW ARE YOU",
    }

    dataset = Dataset.from_list([sample])
    dataset = preprocess_wav2vec2(dataset, wav2vec2)

    result = next(iter(dataset))

    assert list(result.keys()) == ["input_values", "labels", "input_length"], (
        "Incorrect keys"
    )
    assert result["input_values"].shape == (1, sampling_rate)
    assert result["input_length"] == 1.0

    # Repeating tokens get merge
    assert wav2vec2.tokenizer.decode(result["labels"]) == "HELO THERE HOW ARE YOU"


def test_processor_resampling(whisper):
    """Test warning or behavior when audio requires resampling."""
    wrong_sampling_rate = 8000
    audio_array = np.random.randn(int(wrong_sampling_rate * 5.0)).astype(np.float32)

    # Process the audio
    sample = {
        "audio": {"array": audio_array, "sampling_rate": wrong_sampling_rate},
        "text": "HELLO THERE HOW ARE YOU",
    }

    dataset = Dataset.from_list([sample])
    dataset = preprocess_whisper(dataset, whisper)

    result = next(iter(dataset))

    assert list(result.keys()) == ["input_features", "labels", "input_length"], (
        "Incorrect keys"
    )
    assert result["input_length"] == 5.0
