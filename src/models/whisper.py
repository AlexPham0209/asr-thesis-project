import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import ModelOutput


class Whisper(nn.Module):
    def __init__(self, model_id: str = "openai/whisper-tiny"):
        super().__init__()
        print(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)

    def forward(self, input_features=None, labels=None, **kwargs):
        outputs = self.model(input_features=input_features, labels=labels)
        return ModelOutput(loss=outputs.loss, logits=outputs.logits)
