import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, WhisperModel


def mean_pooling(embeddings, attention_mask=None):
    """
    Args:
        input_features: Encoded representation tensor (batch_size, seq_len, hidden_dim)
        attention_mask: Optional mask tensor (batch_size, seq_len)

    Returns:
        Embedding tensor that has been mean-pooled across the seq_len dimension (batch_size, hidden_dim)
    """
    if attention_mask is None:
        return torch.mean(embeddings, dim=1)

    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


class WhisperEmbedding(nn.Module):
    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()

        # Load base Whisper encoder
        self.encoder = WhisperModel.from_pretrained(model_name).get_encoder()
        hidden_dim = self.encoder.config.d_model

        # Linear projection to joint audio-text embedding space
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_features: Log-Mel spectrogram tensor (batch_size, n_mels, time_steps)
            attention_mask: Optional mask tensor (batch_size, time_steps)
        """

        # Retrieving encoded representation of our audio from the Whisper encoder: (batch_size, seq_len, hidden_dim)
        encoder_outputs = self.encoder(input_features=input_features)
        embeddings = encoder_outputs.last_hidden_state

        # Mean pooling across the sequence dimension: (batch_size, hidden_dim)
        if (
            attention_mask is not None
            and attention_mask.shape[1] != embeddings.shape[1]
        ):
            attention_mask = attention_mask[:, ::2]

        pooled = mean_pooling(embeddings, attention_mask)

        # Linearly projecting to joint audio-text embedding dimension then applying Euclidean (L2) normalization: (batch_size, embed_dim)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)


class MathBERTEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
        )

        hidden_dim = getattr(
            self.model.config,
            "hidden_size",
            getattr(self.model.config, "d_model", None),
        )
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self, input_features: dict, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        encoder_outputs = self.model(**input_features)
        embeddings = encoder_outputs.last_hidden_state

        # Apply mean pooling
        pooled = mean_pooling(embeddings, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)


class CustomCLAPModule(nn.Module):
    def __init__(
        self,
        audio_embedding_layer: nn.Module,
        text_embedding_layer: nn.Module,
        init_temperature: float = 0.07,
    ):
        super().__init__()

        self.audio_embedding_layer = audio_embedding_layer
        self.text_embedding_layer = text_embedding_layer

        # Learnable logit scale initialized to CLIP default (log(1/0.07))
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature))
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_features: torch.Tensor,
        text_inputs: torch.Tensor,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ):
        # Compute L2-normalized embeddings
        audio_embeds = self.audio_embedding_layer(
            input_features, attention_mask=audio_mask
        )
        text_embeds = self.text_embedding_layer(text_inputs, attention_mask=text_mask)

        # Compute similarity matrix between audio and text embeddings
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (audio_embeds @ text_embeds.T) * logit_scale

        # Create ground-truth labels
        # Matching pairs should have a dot product/cosine similarity of 1 while every other pair should have a similarity of 0
        # This means that the diagonals of the similarity matrix should be 1
        batch_size = audio_embeds.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Compute symmetric cross-entropy loss
        audio_loss = self.cross_entropy_loss(logits, labels)
        text_loss = self.cross_entropy_loss(logits.T, labels)

        loss = (audio_loss + text_loss) / 2.0
        return logits, loss  #
    
    def get_audio_features(self, input_features, attention_mask=None):
        return self.audio_embedding_layer(
            input_features, attention_mask=attention_mask
        )
        
    def get_text_features(self, input_features, attention_mask=None):
        return self.text_embedding_layer(
            input_features, attention_mask=attention_mask
        )
    
