"""Two serial stages: official Mamba-2 backbone, then LogKV blocks.

Embedding and output weights are shared as in the Mamba-2 baseline. The Mamba
backbone retains its own residuals and final normalization. LogKV has a separate
final normalization, fixed-size Mamba state plus logarithmic hierarchical state,
and no additional convolution. No states are detached at input chunk boundaries.
"""
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.logkv.attention import LogKVBlock
from models.mamba2.modeling import Mamba2LM
from models.mamba2_logkv.configuration import Mamba2LogKVConfig


class Mamba2LogKVLM(Mamba2LM):
    config_class = Mamba2LogKVConfig

    def __init__(self, config):
        super().__init__(config)
        # Use LogKV's native module initialization; do not reinitialize the
        # officially initialized Mamba backbone or tied embedding/head.
        self.logkv_layers = nn.ModuleList([
            LogKVBlock(config.d_model, config.chunk_size, config.d_ff, config.num_heads,
                       config.phase_emb, config.phase_levels, config.learnable_decay,
                       config.gated_attention, config.kv_norm, config.level_amplify,
                       config.v_norm_only, config.self_slot, conv_kernel_size=0)
            for _ in range(config.num_logkv_layers)
        ])
        self.logkv_norm = nn.RMSNorm(config.d_model)
        self.post_init()

    def step(self, input_ids, hidden=None, *, reference=False):
        if hidden is None:
            mamba_state, logkv_state = None, [None] * len(self.logkv_layers)
        else:
            if not isinstance(hidden, dict) or set(hidden) != {'mamba', 'logkv'}:
                raise ValueError("Hybrid state requires mamba and logkv entries")
            mamba_state, logkv_state = hidden['mamba'], hidden['logkv']
            if len(logkv_state) != len(self.logkv_layers):
                raise ValueError("One LogKV state per LogKV layer is required")
        x, mamba_state = self.step_features(input_ids, mamba_state, reference=reference)
        next_logkv = []
        for layer, state in zip(self.logkv_layers, logkv_state):
            x, state = layer.step(x, state)
            next_logkv.append(state)
        return self.lm_head(self.logkv_norm(x)), {'mamba': mamba_state, 'logkv': next_logkv}

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False, **kwargs):
        if past_key_values is None and not use_cache and input_ids.is_cuda:
            # Keep the official Mamba-2 full-sequence training path.
            x = self.backbone(input_ids)
            for layer in self.logkv_layers:
                x = layer(x)
            logits, hidden = self.lm_head(self.logkv_norm(x)), None
        else:
            logits, hidden = self.step(input_ids, past_key_values)
        loss = None
        if labels is not None:
            loss = (nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                                labels.reshape(-1), ignore_index=-100)
                    if (labels != -100).any() else logits.float().sum() * 0)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                     past_key_values=hidden if use_cache else None)
