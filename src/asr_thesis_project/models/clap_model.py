import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel, WhisperModel


class CustomCLAPModule(nn.Module):
    def __init__(
        self,
        audio_embedding_layer: nn.Module,
        text_embedding_layer: nn.Module,
        embed_dim: int,
        init_temperature: float = 0.07,
    ):
        super().__init__()

        self.audio_embedding_layer = audio_embedding_layer
        self.text_embedding_layer = text_embedding_layer

        # Linear projection to joint audio-text embedding space
        self.audio_projection = nn.Linear(
            self.audio_embedding_layer.hidden_dim, embed_dim
        )
        self.text_projection = nn.Linear(
            self.text_embedding_layer.hidden_dim, embed_dim
        )

        # Learnable logit scale initialized to CLIP default (log(1/0.07))
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature))
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        audio_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ):
        # Compute L2-normalized audio and text embeddings
        audio_embeds = self.get_audio_features(input_features=audio_inputs, audio_mask=audio_mask)
        text_embeds = self.get_text_features(input_features=text_inputs, text_mask=text_mask)

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
        return logits, loss

    def get_audio_features(self, input_features, attention_mask=None):
        audio_embeds = self.audio_embedding_layer(input_features, attention_mask=attention_mask)
        audio_embeds = self.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)
        return audio_embeds

    def get_text_features(self, input_features, attention_mask=None):
        text_embeds = self.text_embedding_layer(input_features, attention_mask=attention_mask)
        text_embeds = self.text_projection(text_embeds)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return text_embeds
