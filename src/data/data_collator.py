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

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
