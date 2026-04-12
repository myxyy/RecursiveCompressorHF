import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor import RecursiveCompressor


class RecursiveCompressorLM(PreTrainedModel):
    config_class = RecursiveCompressorConfig

    def __init__(self, config: RecursiveCompressorConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            RecursiveCompressor(config.d_model, config.num_heads, config.d_ff, config.chunk_size, config.compress_size)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def step(self, input_ids, hidden):
        x = self.embedding(input_ids)
        if hidden is None:
            hidden = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, hidden[i] = layer.step(x, hidden[i])
        x = self.norm(x)
        logits = self.head(x)
        return logits, hidden

    def forward(self, input_ids, labels=None, **kwargs):
        logits, _ = self.step(input_ids, None)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.float().view(-1, self.config.vocab_size), labels.view(-1))
        return CausalLMOutput(loss=loss, logits=logits)

    def predict(self, input_ids, hidden):
        logits, hidden = self.step(input_ids.unsqueeze(-1), hidden)
        return logits.squeeze(1), hidden
