import torch
import torch.nn as nn


class CustomCTCModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        # A simple placeholder encoder-decoder for illustration
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_values, labels=None):
        # input_values: (batch, time, features)
        x, _ = self.encoder(input_values)
        logits = self.classifier(x)

        output = {"logits": logits}
        if labels is not None:
            # You can calculate CTC loss here to match HF API behavior
            pass

        return output
