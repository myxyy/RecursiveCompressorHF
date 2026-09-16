"""LogKV layers partitioned across devices, with ordinary HF weight names."""
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import PreTrainedModel

from configuration_logkv import LogKVConfig
from logkv import LogKVBlock
from logkv_lm import LogKVLM


class LogKVLMPipelineStage(PreTrainedModel):
    config_class = LogKVConfig

    def __init__(self, config, layer_start, layer_end, is_first, is_last):
        super().__init__(config)
        if not 0 <= layer_start < layer_end <= config.num_layers:
            raise ValueError('Invalid pipeline layer range')
        self.stage_info = dict(layer_start=layer_start, layer_end=layer_end,
                               is_first=is_first, is_last=is_last)
        self.is_first, self.is_last = is_first, is_last
        if is_first:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Global indices keep state_dict names identical to LogKVLM. Only
        # local layers are constructed: no full model allocation on each GPU.
        self.layers = nn.ModuleDict({str(i): LogKVBlock(
            config.d_model, config.chunk_size, config.d_ff, config.num_heads,
            config.phase_emb, config.phase_levels, config.learnable_decay,
            config.gated_attention, config.kv_norm, config.level_amplify,
            config.v_norm_only, config.self_slot, config.conv_kernel_size)
            for i in range(layer_start, layer_end)})
        if is_last:
            self.norm = nn.RMSNorm(config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Same HF initialization policy as the ordinary LM (including conv).
        self.post_init()

    def forward(self, x):
        if self.is_first:
            x = self.embedding(x)
        for layer in self.layers.values():
            x = layer(x)
        return self.head(self.norm(x)) if self.is_last else x

    def step(self, x, hidden=None, *, single_token=False):
        """Local cache for distributed sample generation only."""
        if self.is_first:
            x = self.embedding(x)
        hidden = [None] * len(self.layers) if hidden is None else list(hidden)
        for i, layer in enumerate(self.layers.values()):
            fn = layer.predict if single_token else layer.step
            x, hidden[i] = fn(x, hidden[i])
        if self.is_last:
            x = self.head(self.norm(x))
        return x, hidden

    @staticmethod
    def split_config(num_layers, num_stages, layer_counts=None):
        if not 1 <= num_stages <= num_layers:
            raise ValueError('Every pipeline stage needs at least one layer')
        counts = (list(layer_counts) if layer_counts is not None else
                  [num_layers // num_stages + (i < num_layers % num_stages)
                   for i in range(num_stages)])
        if len(counts) != num_stages or sum(counts) != num_layers or any(c < 1 for c in counts):
            raise ValueError('stage-layer-split must have one positive count per rank and sum to num-layers')
        start, stages = 0, []
        for rank, count in enumerate(counts):
            stages.append(dict(layer_start=start, layer_end=start + count,
                               is_first=rank == 0, is_last=rank == num_stages - 1))
            start += count
        return stages

    def load_from_full_model(self, state):
        self.load_state_dict({key: state[key] for key in self.state_dict()}, strict=True)

    def load_model_directory(self, model_dir):
        """Read only this stage's tensors from HF safetensors (also sharded)."""
        wanted = set(self.state_dict())
        local = {}
        for path in sorted(Path(model_dir).glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in wanted.intersection(f.keys()):
                    if key in local:
                        raise ValueError(f'Duplicate tensor in model directory: {key}')
                    local[key] = f.get_tensor(key)
        if set(local) != wanted:
            raise ValueError(f'Missing safetensors weights in {model_dir}: {sorted(wanted - set(local))}')
        self.load_state_dict(local, strict=True)


def export_model(checkpoint_dir, config, tokenizer, stage_infos):
    """Rank 0 only. Reassemble on CPU, never on a training GPU.

    Per-stage optimizer files are deliberately separate, so export reads only
    model weights. A meta model avoids allocating a second full set of weights.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state = {}
    for rank, expected_info in enumerate(stage_infos):
        saved = torch.load(checkpoint_dir / f'stage_{rank}.pt', map_location='cpu', weights_only=True)
        if saved['stage_info'] != expected_info:
            raise ValueError(f'Wrong layout in stage {rank}')
        overlap = state.keys() & saved['model'].keys()
        if overlap:
            raise ValueError(f'Duplicate stage weights: {sorted(overlap)}')
        state.update(saved['model'])
    with torch.device('meta'):
        model = LogKVLM(config)
    model.load_state_dict(state, strict=True, assign=True)
    model_dir = checkpoint_dir / 'model'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir
