import math
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
        self.compressor_query = nn.Parameter(torch.randn(config.compress_size, config.d_model))
        self.layers = nn.ModuleList([
            RecursiveCompressor(config.d_model, config.num_heads, config.d_ff, config.chunk_size, config.compress_size)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def _num_queries(self):
        """Maximum number of compressor queries needed for recursion depth + 1."""
        return math.ceil(math.log(65536) / math.log(self.config.chunk_size)) + 1

    def _make_xs(self, x):
        """Create xs list: [data, q, q, q, ...] with enough queries for max recursion."""
        batch_size = x.size(0)
        n = self._num_queries()
        q = self.compressor_query[None, :, :].expand(batch_size, -1, -1)
        return [x] + [q for _ in range(n)]

    def step(self, input_ids, hidden):
        x = self.embedding(input_ids)
        xs = self._make_xs(x)
        if hidden is None:
            hidden = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            xs, hidden[i] = layer.step(xs, hidden[i])
        x = xs[0]
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
