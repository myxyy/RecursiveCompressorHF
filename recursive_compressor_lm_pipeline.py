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

        self.layers = nn.ModuleList([
            RecursiveCompressor(config.d_model, config.num_heads, config.d_ff, config.chunk_size,
                                config.compress_size, getattr(config, "retrieve_size", 4))
            for _ in range(self.num_local_layers)
        ])

        if is_last:
            self.norm = nn.RMSNorm(config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _num_queries(self):
        return math.ceil(math.log(65536) / math.log(self.config.chunk_size)) + 1

    def _make_xs(self, x):
        """Queries all start as None; each layer's first stage derives its
        per-chunk compressor query from the data (see RecursiveCompressorAttention)."""
        n = self._num_queries()
        return [x] + [None for _ in range(n)]

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
        """Pack the xs list into a single tensor for inter-stage transfer.

        xs[0]: (batch, seq_len, d_model) - main data.
        xs[k] (k>=1): per-chunk query (batch, S_k, compress_size, d_model) or None,
            where S_k = seq_len // chunk_size**k (None once S_k hits 0).

        Each query slot is flattened to (batch, S_k*compress_size, d_model), padded
        to seq_len, and stacked. None slots become zero padding; the receiver
        reconstructs which slots are None from seq_len (the structure is
        deterministic), so the zeros are never consumed."""
        data = xs[0]  # (batch, seq_len, d_model)
        batch_size, seq_len, d_model = data.size()

        padded = [data]
        for slot in xs[1:]:
            if slot is None:
                padded.append(torch.zeros(batch_size, seq_len, d_model, device=data.device, dtype=data.dtype))
                continue
            flat = slot.reshape(batch_size, -1, d_model)  # (batch, S_k*compress_size, d_model)
            pad_len = seq_len - flat.size(1)
            if pad_len > 0:
                flat = torch.cat([flat, torch.zeros(batch_size, pad_len, d_model, device=data.device, dtype=data.dtype)], dim=1)
            padded.append(flat)

        return torch.stack(padded, dim=1)  # (batch, 1+num_queries, seq_len, d_model)

    def _unpack_xs(self, packed):
        """Inverse of _pack_xs. Recomputes each slot's length S_k from seq_len
        and reconstructs None for slots where S_k == 0."""
        # packed: (batch, 1+num_queries, seq_len, d_model)
        batch_size, num_slots, seq_len, d_model = packed.size()
        cs = self.config.compress_size
        chunk = self.config.chunk_size

        xs = [packed[:, 0]]  # data: (batch, seq_len, d_model)
        s = seq_len
        for k in range(1, num_slots):
            s = s // chunk  # S_k
            if s > 0:
                flat = packed[:, k, :s * cs, :]  # (batch, S_k*compress_size, d_model)
                xs.append(flat.reshape(batch_size, s, cs, d_model))
            else:
                xs.append(None)
        return xs

    @staticmethod
    def split_config(num_layers, num_stages, layer_counts=None):
        """Divide layers across stages. Returns list of config dicts.

        layer_counts: optional explicit per-stage layer counts (list of ints
        summing to num_layers). Use this to skew layers toward LATER stages:
        under 1F1B, stage r holds (num_stages - r) in-flight microbatch
        activations, so early stages need fewer layers to balance VRAM.
        None = even split (remainder added to the first stages)."""
        if layer_counts is not None:
            assert len(layer_counts) == num_stages, \
                f"layer_counts has {len(layer_counts)} entries, expected {num_stages}"
            assert sum(layer_counts) == num_layers, \
                f"layer_counts sums to {sum(layer_counts)}, expected {num_layers}"
            assert all(c >= 1 for c in layer_counts), "every stage needs >= 1 layer"
            counts = list(layer_counts)
        else:
            base = num_layers // num_stages
            remainder = num_layers % num_stages
            counts = [base + (1 if i < remainder else 0) for i in range(num_stages)]

        stages = []
        start = 0
        for i, count in enumerate(counts):
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
            elif self.is_first and key == "embedding.weight":
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
