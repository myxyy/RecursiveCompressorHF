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

    def step(self, input_ids, hidden):
        x = self.embedding(input_ids)
        if hidden is None:
            hidden = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, hidden[i] = layer.step(x, hidden[i])
        x = self.norm(x)
        logits = self.head(x)
        return logits, hidden

    def forward(self, input_ids):
        logits, _ = self.step(input_ids, None)
        return logits

    def predict(self, input_ids, hidden):
        logits, hidden = self.step(input_ids.unsqueeze(-1), hidden)
        return logits.squeeze(1), hidden