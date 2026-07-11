import pytest
import torch
from transformers import AutoProcessor

from src.data.data_collator import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)


@pytest.fixture(scope="module")
def wav2vec2():
    return AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")


@pytest.fixture(scope="module")
def whisper():
    return AutoProcessor.from_pretrained("openai/whisper-tiny")


def test_ctc_collate(wav2vec2):
    collator = DataCollatorCTCWithPadding(wav2vec2)

    sample1 = {
        "input_values": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([1, 4, 5, 6]),
        "input_length": 1.0,
    }

    sample2 = {
        "input_values": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        "labels": torch.tensor([1, 2]),
        "input_length": 1.0,
    }

    batch = [sample1, sample2]
    collated = collator(batch)

    pad_token = wav2vec2.tokenizer.pad_token_id
    input_values = collated["input_values"]
    correct_values = torch.tensor(
        [[1, 2, 3, 4, pad_token, pad_token, pad_token], [1, 2, 3, 4, 5, 6, 7]]
    )

    labels = collated["labels"]
    correct_labels = torch.tensor(
        [
            [1, 4, 5, 6],
            [1, 2, -100, -100],
        ]
    )

    assert input_values.shape == (2, 7)
    assert torch.all(torch.eq(input_values, correct_values))

    assert labels.shape == (2, 4)
    assert torch.all(torch.eq(labels, correct_labels))


def test_seq2seq_collate(whisper):
    collator = DataCollatorSpeechSeq2SeqWithPadding(whisper)

    bos_token = whisper.tokenizer.bos_token_id

    sample1 = {
        "input_features": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        "labels": torch.tensor([bos_token, 1, 4, 5, 6]),
        "input_length": 1.0,
    }

    sample2 = {
        "input_features": torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]),
        "labels": torch.tensor([bos_token, 1, 2]),
        "input_length": 1.0,
    }

    batch = [sample1, sample2]
    collated = collator(batch)

    pad_token = whisper.tokenizer.pad_token_id
    input_values = collated["input_features"]
    correct_values = torch.tensor(
        [
            [[1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0]],
            [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]],
        ]
    )

    labels = collated["labels"]
    correct_labels = torch.tensor(
        [
            [1, 4, 5, 6],
            [1, 2, -100, -100],
        ]
    )

    assert input_values.shape == (2, 2, 7)
    assert torch.all(torch.eq(input_values, correct_values))

    assert labels.shape == (2, 4)
    assert torch.all(torch.eq(labels, correct_labels))
