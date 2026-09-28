"""x = LogKV embedding + tanh(gate) * normalized Mamba-2 features.

The gate is a zero-initialized, input-independent vector of channel scales.
LogKV embedding, Conv/Attention/FFN blocks, final norm and untied head remain
unchanged. LogKV is constructed first, so a zero gate exactly recovers the
same-seed standalone LogKV, not merely its topology. The Mamba branch has its
own embedding and official initialization; no pretrained weights are used.
"""
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.logkv.modeling import LogKVLM
from models.mamba2.modeling import Mamba2LM
from models.mamba2_logkv_gated.configuration import GatedMambaLogKVConfig


class GatedMambaLogKVLM(LogKVLM):
    config_class = GatedMambaLogKVConfig

    def __init__(self, config):
        super().__init__(config)
        self.mamba = Mamba2LM(config.mamba_config())
        self.mamba_gate = nn.Parameter(torch.zeros(config.d_model))
        self.post_init()

    def _fuse(self, input_ids, features):
        x = self.embedding(input_ids)
        return x + torch.tanh(self.mamba_gate).to(x.dtype) * features.to(x.dtype)

    def step(self, input_ids, hidden=None, *, reference=False):
        if hidden is None:
            mamba_state, logkv_state = None, [None] * len(self.layers)
        else:
            if not isinstance(hidden, dict) or set(hidden) != {'mamba', 'logkv'}:
                raise ValueError("Expected mamba and logkv state entries")
            mamba_state, logkv_state = hidden['mamba'], hidden['logkv']
            if len(logkv_state) != len(self.layers):
                raise ValueError("One state per LogKV layer is required")
        features, mamba_state = self.mamba.step_features(input_ids, mamba_state, reference=reference)
        x = self._fuse(input_ids, features)
        next_logkv = []
        for layer, state in zip(self.layers, logkv_state):
            x, state = layer.step(x, state)
            next_logkv.append(state)
        return self.head(self.norm(x)), {'mamba': mamba_state, 'logkv': next_logkv}

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False, **kwargs):
        if past_key_values is None and not use_cache and input_ids.is_cuda:
            x = self._fuse(input_ids, self.mamba.backbone(input_ids))
            for layer in self.layers:
                x = layer(x)
            logits, hidden = self.head(self.norm(x)), None
        else:
            logits, hidden = self.step(input_ids, past_key_values)
        loss = None
        if labels is not None:
            loss = (nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                                labels.reshape(-1), ignore_index=-100)
                    if (labels != -100).any() else logits.float().sum() * 0)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                     past_key_values=hidden if use_cache else None)

    def predict(self, input_ids, hidden=None):
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.shape[1] != 1:
            raise ValueError("predict expects one token per example")
        out, hidden = self.step(input_ids, hidden)
        return out[:, 0], hidden

    def gate_metrics(self):
        gate = torch.tanh(self.mamba_gate.detach().float())
        return dict(min=gate.min().item(), max=gate.max().item(),
                    mean=gate.mean().item(), abs_mean=gate.abs().mean().item(),
                    values=gate.cpu().tolist())
