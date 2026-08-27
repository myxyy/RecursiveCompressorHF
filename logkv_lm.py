import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_logkv import LogKVConfig
from logkv import LogKVBlock


class LogKVLM(PreTrainedModel, GenerationMixin):
    config_class = LogKVConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: LogKVConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            LogKVBlock(config.d_model, config.chunk_size, config.d_ff, config.num_heads,
                       config.phase_emb, config.phase_levels)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def step(self, input_ids, hidden=None):
        """Chunked sequential forward over all layers.

        hidden: list of per-layer LogKV hidden states ((levels, offset)
        tuples), or None to start. Concatenating step() logits over any split
        of the input reproduces forward(). Returns (logits, hidden)."""
        x = self.embedding(input_ids)
        if hidden is None:
            hidden = [None] * len(self.layers)
        hidden = list(hidden)
        for i, layer in enumerate(self.layers):
            x, hidden[i] = layer.step(x, hidden[i])
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
        attention_mask:  ignored (the hierarchical kv attention is internal).
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
        if model_kwargs.get("past_key_values", None) is None:
            model_kwargs["past_key_values"] = None
        return False  # cache is NOT prepared by HF; we manage it ourselves

    def predict(self, input_ids, hidden=None):
        """Single-token inference: input_ids is (batch_size,). Returns
        (logits (batch_size, vocab_size), new_hidden)."""
        logits, hidden = self.step(input_ids.unsqueeze(-1), hidden)
        return logits.squeeze(1), hidden
