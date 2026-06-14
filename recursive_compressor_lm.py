import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor import RecursiveCompressor


class RecursiveCompressorLM(PreTrainedModel, GenerationMixin):
    config_class = RecursiveCompressorConfig
    # Used by HF generate's caching machinery. The actual cache type we expose
    # is opaque (a list of per-layer hidden states), so this is a marker only.
    supports_gradient_checkpointing = False

    def __init__(self, config: RecursiveCompressorConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            RecursiveCompressor(config.d_model, config.num_heads, config.d_ff, config.chunk_size, config.compress_size)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def _num_queries(self):
        """Maximum number of compressor queries needed for recursion depth + 1."""
        return math.ceil(math.log(65536) / math.log(self.config.chunk_size)) + 1

    def _make_xs(self, x):
        """Create xs list: [data, q, q, q, ...] with enough queries for max recursion."""
        batch_size = x.size(0)
        n = self._num_queries()
        return [x] + [None for _ in range(n)]

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

    def forward(
        self,
        input_ids,
        labels=None,
        past_key_values=None,
        use_cache=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        past_key_values: opaque hidden state from a previous forward call
                         (a list of per-layer hidden states), or None.
        use_cache:       if True, return updated hidden state in
                         `past_key_values` of the output. Set to True
                         automatically by HF `generate()`.
        attention_mask:  ignored (the recursive chunk attention is internal).
        """
        logits, hidden = self.step(input_ids, past_key_values)

        loss = None
        if labels is not None:
            flat_targets = labels.view(-1)
            flat_logits = logits.float().view(-1, self.config.vocab_size)
            # Guard against all-PAD-label samples (would give 0/0 = NaN).
            if (flat_targets != -100).sum() == 0:
                loss = flat_logits.sum() * 0.0
            else:
                loss = nn.CrossEntropyLoss()(flat_logits, flat_targets)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=hidden if use_cache else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """Called by HF `generate()` once per generation step.
        On the first step past_key_values is None, so we feed the full prompt.
        After that, only the most recently sampled token is fed; the rest of
        the context is encoded in our hidden state."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def can_generate(self):
        # PreTrainedModel returns False unless `prepare_inputs_for_generation`
        # is overridden — we just made sure of that, so explicitly enable.
        return True

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length,
    ):
        """Disable HF's automatic Cache class wrapping. Our `past_key_values`
        is an opaque per-layer hidden state list, not a (K, V) cache."""
        # Make sure past_key_values stays None (not wrapped in DynamicCache)
        if model_kwargs.get("past_key_values", None) is None:
            model_kwargs["past_key_values"] = None
        return False  # cache is NOT prepared by HF; we manage it ourselves

    def predict(self, input_ids, hidden):
        logits, hidden = self.step(input_ids.unsqueeze(-1), hidden)
        return logits.squeeze(1), hidden
