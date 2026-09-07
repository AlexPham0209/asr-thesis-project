import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, WhisperModel


class WhisperEmbedding(nn.Module):
    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        
        # Load base Whisper encoder
        self.encoder = WhisperModel.from_pretrained(model_name).get_encoder()
        hidden_dim = self.encoder.config.d_model
        
        # Linear projection to joint audio-text embedding space
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_features: Log-Mel spectrogram tensor (batch_size, n_mels, time_steps)
            attention_mask: Optional mask tensor (batch_size, time_steps)
        """
        
        # Retrieving encoded representation of our audio from the Whisper encoder: (batch_size, seq_len, hidden_dim) 
        encoder_outputs = self.encoder(input_features=input_features)
        last_hidden_state = encoder_outputs.last_hidden_state 
        
        # Mean pooling across the sequence dimension: (batch_size, hidden_dim)
        if attention_mask is not None:
            # Dynamically resize mask to match Whisper encoder output length
            target_len = last_hidden_state.shape[1]
            mask = F.interpolate(
                attention_mask.unsqueeze(1).float(), 
                size=target_len, 
                mode='nearest'
            ).squeeze(1).unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            pooled = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            pooled = torch.mean(last_hidden_state, dim=1)
        
        # Linearly projecting to joint audio-text embedding dimension then applying Euclidean (L2) normalization: (batch_size, embed_dim)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)
    
    
class MathBERTEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.model = AutoModel.from_pretrained('math-similarity/Bert-MLM_arXiv-MP-class_zbMath')
        
        hidden_dim = getattr(self.model.config, "hidden_size", getattr(self.model.config, "d_model", None))
        self.projection = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        model_output = self.model(**input_features)
        
        # Apply mean pooling
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)


class CustomCLAPModule(nn.Module):
    def __init__(self, audio_embedding_layer: nn.Module, text_embedding_layer: nn.Module, init_temperature: float = 0.07):
        super().__init__()
        
        self.audio_embedding_layer = audio_embedding_layer
        self.text_embedding_layer = text_embedding_layer
        
        # Learnable logit scale initialized to CLIP default (log(1/0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature)))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def forward(self, input_features: torch.Tensor, text_inputs: torch.Tensor, audio_mask: torch.Tensor = None, text_mask: torch.Tensor = None):
        # Compute L2-normalized embeddings
        audio_embeds = self.audio_embedding_layer(input_features, attention_mask=audio_mask)
        text_embeds = self.text_embedding_layer(text_inputs, attention_mask=text_mask)
        
        # Compute similarity matrix between audio and text embeddings
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (audio_embeds @ text_embeds.T) * logit_scale
        
        # Create ground-truth labels on correct device
        batch_size = audio_embeds.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Compute symmetric cross-entropy loss
        audio_loss = self.cross_entropy_loss(logits, labels)
        text_loss = self.cross_entropy_loss(logits.T, labels)
        
        loss = (audio_loss + text_loss) / 2.0
        return logits, loss