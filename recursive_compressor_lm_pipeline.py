import math
import torch
import torch.nn as nn
from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor import RecursiveCompressor


class RecursiveCompressorLMPipelineStage(nn.Module):
    """A single pipeline stage of RecursiveCompressorLM.

    First stage owns: embedding + compressor_query + first N layers
    Middle stages own: some layers
    Last stage owns: last N layers + norm + head
    """

    def __init__(self, config: RecursiveCompressorConfig, layer_start, layer_end, is_first, is_last):
        super().__init__()
        self.config = config
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = is_first
        self.is_last = is_last
        self.num_local_layers = layer_end - layer_start

        if is_first:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.compressor_query = nn.Parameter(torch.randn(config.compress_size, config.d_model))

        self.layers = nn.ModuleList([
            RecursiveCompressor(config.d_model, config.num_heads, config.d_ff, config.chunk_size, config.compress_size)
            for _ in range(self.num_local_layers)
        ])

        if is_last:
            self.norm = nn.LayerNorm(config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _num_queries(self):
        return math.ceil(math.log(65536) / math.log(self.config.chunk_size)) + 1

    def _make_xs(self, x):
        batch_size = x.size(0)
        n = self._num_queries()
        q = self.compressor_query[None, :, :].expand(batch_size, -1, -1)
        return [x] + [q for _ in range(n)]

    def forward(self, x):
        if self.is_first:
            x = self.embedding(x)
            xs = self._make_xs(x)
        else:
            xs = self._unpack_xs(x)

        for layer in self.layers:
            xs, _ = layer.step(xs, None)

        if self.is_last:
            x = xs[0]
            x = self.norm(x)
            logits = self.head(x)
            return logits
        else:
            return self._pack_xs(xs)

    def _pack_xs(self, xs):
        """Pack list of tensors into a single tensor for inter-stage transfer.
        xs[0]: (batch, seq_len, d_model), xs[1:]: (batch, compress_size, d_model)
        Pack by padding xs[1:] to seq_len and stacking along a new dim."""
        data = xs[0]  # (batch, seq_len, d_model)
        batch_size, seq_len, d_model = data.size()
        queries = xs[1:]  # each (batch, compress_size, d_model)

        padded = [data]
        for q in queries:
            # Pad compress_size -> seq_len
            pad_len = seq_len - q.size(1)
            padded_q = torch.cat([q, torch.zeros(batch_size, pad_len, d_model, device=q.device, dtype=q.dtype)], dim=1)
            padded.append(padded_q)

        return torch.stack(padded, dim=1)  # (batch, 1+num_queries, seq_len, d_model)

    def _unpack_xs(self, packed):
        """Unpack tensor back to list of tensors."""
        # packed: (batch, 1+num_queries, seq_len, d_model)
        data = packed[:, 0]  # (batch, seq_len, d_model)
        queries = []
        compress_size = self.config.compress_size
        for i in range(1, packed.size(1)):
            queries.append(packed[:, i, :compress_size, :])  # (batch, compress_size, d_model)
        return [data] + queries

    @staticmethod
    def split_config(num_layers, num_stages):
        """Divide layers across stages. Returns list of config dicts."""
        base = num_layers // num_stages
        remainder = num_layers % num_stages
        stages = []
        start = 0
        for i in range(num_stages):
            count = base + (1 if i < remainder else 0)
            stages.append({
                "layer_start": start,
                "layer_end": start + count,
                "is_first": (i == 0),
                "is_last": (i == num_stages - 1),
            })
            start += count
        return stages

    def load_from_full_model(self, full_state_dict):
        """Load weights from a full RecursiveCompressorLM state dict."""
        local_state = {}
        for key, value in full_state_dict.items():
            if key.startswith("layers."):
                parts = key.split(".", 2)
                global_idx = int(parts[1])
                if self.layer_start <= global_idx < self.layer_end:
                    local_idx = global_idx - self.layer_start
                    local_key = f"layers.{local_idx}.{parts[2]}"
                    local_state[local_key] = value
            elif self.is_first and key in ("embedding.weight", "compressor_query"):
                local_state[key] = value
            elif self.is_last and (key.startswith("norm.") or key.startswith("head.")):
                local_state[key] = value
        self.load_state_dict(local_state)

    @staticmethod
    def reconstruct_full_state_dict(gathered):
        """Reconstruct full model state dict from gathered pipeline stages.
        gathered: list of (rank, stage_info, state_dict) tuples."""
        full_state = {}
        gathered = sorted(gathered, key=lambda x: x[0])
        for _rank, stage_info, state_dict in gathered:
            layer_start = stage_info["layer_start"]
            for key, value in state_dict.items():
                if key.startswith("layers."):
                    parts = key.split(".", 2)
                    local_idx = int(parts[1])
                    global_idx = local_idx + layer_start
                    full_key = f"layers.{global_idx}.{parts[2]}"
                    full_state[full_key] = value
                else:
                    full_state[key] = value
        return full_state
