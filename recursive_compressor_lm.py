import torch
import torch.nn as nn
from recursive_compressor import RecursiveCompressor


class RecursiveCompressorLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, chunk_size, compress_size, num_layers):
        super(RecursiveCompressorLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            RecursiveCompressor(d_model, num_heads, d_ff, chunk_size, compress_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def predict(self, input_ids, hidden: list[list[tuple[None | torch.Tensor, None | torch.Tensor]] | None] | None):
        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_layer = hidden[i] if hidden is not None else None
            x, hidden_layer = layer.predict(x, hidden_layer)
            if hidden is not None:
                hidden[i] = hidden_layer
        x = self.norm(x)
        logits = self.head(x)
        return logits, hidden