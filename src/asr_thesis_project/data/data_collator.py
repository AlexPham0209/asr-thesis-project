import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import AutoProcessor


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: list[dict[str, Union[list[int], torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    bos_token_id: int
    padding: Union[bool, str] = "longest"
    
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]

        # Audio feature extractor expects True for dynamic padding, not "longest"
        audio_padding = True if self.padding == "longest" else self.padding

        batch = self.processor.feature_extractor.pad(
            input_features, padding=audio_padding, return_tensors="pt"
        )

        # Tokenizer handles "longest" perfectly fine
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorAudioLMWithPadding:
    """Batch collator for decoder-only audio LLMs (Qwen2-Audio, Voxtral, ...).

    Rows come from preprocess_multimodal.preprocess_speech2latex: raw waveform +
    chat-template strings + prompt_length. The processor is run here, per batch,
    so audio-token expansion, mel features and text padding all come from one
    call and stay consistent. Requires TrainingArguments(remove_unused_columns=False).
    """

    processor: Any
    sampling_rate: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [np.asarray(f["audio"], dtype=np.float32) for f in features]
        texts = [f["full_text"] for f in features]

        # Right padding so `labels[i, :prompt_length]` indexes the real prompt.
        self.processor.tokenizer.padding_side = "right"
        batch = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, f in enumerate(features):
            labels[i, : int(f["prompt_length"])] = -100  # loss only on the answer
        batch["labels"] = labels

        return batch
