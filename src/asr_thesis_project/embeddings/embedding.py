from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    BertTokenizer,
    WhisperFeatureExtractor,
)

from asr_thesis_project.models.embeddings import (
    MathBERTEmbeddingModule,
    SentenceEmbeddingModule,
    WhisperEmbeddingModule,
)


class BaseEmbedding(ABC):
    @abstractmethod
    def embedding(self, input) -> list:
        pass

    def __call__(self, input: list) -> list:
        return self.embedding(input)


class CLAPEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        sampling_rate: int = 48000,
        device: str = None,
    ):
        self.sampling_rate = sampling_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.eval()

    def embedding(self, input) -> list:
        inputs = self.feature_extractor(
            input, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            audio_features = self.model.get_audio_features(**inputs)
            audio_features = F.normalize(audio_features, p=2, dim=-1)

        return audio_features.cpu().tolist()


class WhisperEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        sampling_rate: int = 16000,
        device: str = None,
    ):
        self.sampling_rate = sampling_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Fixed: Assign to self.model and send to device
        self.model = WhisperEmbeddingModule(model_name).to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.model.eval()

    def embedding(self, input) -> list:
        inputs = self.feature_extractor(
            input,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            audio_features = self.model(
                input_features=inputs.input_features,
                attention_mask=inputs.attention_mask,
            )

            # Normalize embeddings for Cosine distance
            audio_features = F.normalize(audio_features, p=2, dim=-1)
            embeddings = audio_features.cpu().tolist()

        return embeddings


class MathBERTEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "tbs17/MathBERT",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MathBERTEmbeddingModule(model_name=model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def embedding(self, input) -> list:
        if isinstance(input, str):
            input = [input]
        input = [text.lower() for text in input]
        inputs = self.tokenizer(
            input,
            padding=True,  # batch of different-length sentences -> must pad to tensorize
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            features = self.model(**inputs)
            features = F.normalize(features, p=2, dim=-1)

        return features.cpu().tolist()

class SentenceEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        pooling: str = "cls",
        prefix: str = "",
        max_length: int = 512,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceEmbeddingModule(model_name=model_name, pooling=pooling).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prefix = prefix
        self.max_length = max_length
        self.model.eval()

    def embedding(self, input) -> list:
        if isinstance(input, str):
            input = [input]
        texts = [f"{self.prefix}{text}" for text in input]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            features = self.model(**inputs)
            features = F.normalize(features, p=2, dim=-1)

        return features.cpu().tolist()
