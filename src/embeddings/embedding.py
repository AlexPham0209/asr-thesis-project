from abc import ABC, abstractmethod
import torch
from transformers import AutoFeatureExtractor, AutoModel
import torch.nn.functional as F


class BaseEmbedding(ABC):
    @abstractmethod
    def embedding(self, input) -> list:
        pass

    def __call__(self, input: list) -> list:
        return self.embedding(input)


class CLAPEmbedding(BaseEmbedding):
    def __init__(self, sampling_rate: int = 48000):
        self.sampling_rate = sampling_rate
        self.model = AutoModel.from_pretrained(
            "laion/clap-htsat-unfused", device_map="auto"
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "laion/clap-htsat-unfused"
        )
        self.model.eval()

    def embedding(self, input) -> list:
        inputs = self.feature_extractor(
            input, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            audio_features = self.model.get_audio_features(**inputs)
            audio_features = F.normalize(audio_features, p=2, dim=-1)

        return audio_features.cpu().tolist()
