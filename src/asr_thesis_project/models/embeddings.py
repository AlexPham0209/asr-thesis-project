import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel, WhisperModel

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


class WhisperEmbeddingModule(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        # Load base Whisper encoder
        self.encoder = WhisperModel.from_pretrained(model_name).get_encoder()
        self.hidden_dim = self.encoder.config.d_model

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
        return pooled


class MathBERTEmbeddingModule(nn.Module):
    def __init__(self, model_name: str = "tbs17/MathBERT"):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None, 
        token_type_ids: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        embeddings = encoder_outputs.last_hidden_state

        # Mean-pool using the explicitly passed attention_mask
        return mean_pooling(embeddings, attention_mask)