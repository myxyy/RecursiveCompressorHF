"""Official Mamba-2 backbone with HF storage and LogKV-compatible chunk API."""
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.mamba2.configuration import Mamba2Config
from models.mamba2.streaming import mixer_step, rms_norm


class Mamba2LM(PreTrainedModel, GenerationMixin):
    config_class = Mamba2Config
    _tied_weights_keys = {"lm_head.weight": "backbone.embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        except ImportError as exc:
            raise ImportError("Mamba-2 requires: uv sync --extra mamba2") from exc
        # Upstream constructs AND initializes the complete model, including dt/A,
        # residual rescaling and tied embeddings. Do not post_init/reinitialize it.
        official = MambaLMHeadModel(config.upstream_config())
        self.backbone = official.backbone
        self.lm_head = official.lm_head
        self.post_init()

    def _init_weights(self, module):
        # HF needs post_init for tied-weight bookkeeping, but the official
        # constructor already performed all initialization and rescaling.
        pass

    def get_input_embeddings(self):
        return self.backbone.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def step(self, input_ids, hidden=None, *, reference=False):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape (batch, positive length)")
        if hidden is None:
            hidden = [None] * len(self.backbone.layers)
        if len(hidden) != len(self.backbone.layers):
            raise ValueError("One state per Mamba-2 layer is required")
        x = self.backbone.embedding(input_ids)
        residual, states = None, []
        for block, state in zip(self.backbone.layers, hidden):
            residual = x if residual is None else residual + x
            if x.is_cuda and not reference:
                x = block.norm(residual.to(block.norm.weight.dtype))
            else:
                x = rms_norm(residual.to(block.norm.weight.dtype), block.norm.weight, block.norm.eps)
            if residual.dtype != torch.float64:
                residual = residual.float()
            x, state = mixer_step(block.mixer, x, state, reference=reference)
            states.append(state)
        x = (residual + x).to(self.backbone.norm_f.weight.dtype)
        if x.is_cuda and not reference:
            x = self.backbone.norm_f(x)
        else:
            x = rms_norm(x, self.backbone.norm_f.weight, self.backbone.norm_f.eps)
        return self.lm_head(x), states

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False, **kwargs):
        if past_key_values is None and not use_cache and input_ids.is_cuda:
            # Actual upstream model path for ordinary GPU training, not a rewrite.
            logits = self.lm_head(self.backbone(input_ids))
            hidden = None
        else:
            logits, hidden = self.step(input_ids, past_key_values)
        loss = None
        if labels is not None:
            loss = (F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                    labels.reshape(-1), ignore_index=-100)
                    if (labels != -100).any() else logits.float().sum() * 0)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                     past_key_values=hidden if use_cache else None)

    def predict(self, input_ids, hidden=None):
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.shape[1] != 1:
            raise ValueError("predict expects one token per batch element")
        logits, hidden = self.step(input_ids, hidden)
        return logits[:, 0], hidden

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return dict(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

    def _prepare_cache_for_generation(self, generation_config, model_kwargs,
                                     generation_mode, batch_size, max_cache_length):
        model_kwargs.setdefault('past_key_values', None)
        return False  # Our state is (convolution history, SSM state), not KV pairs.
