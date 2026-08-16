import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.query_linear = nn.Linear(d_model, d_model)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # Open the gate at init: with default (near-zero) init the sigmoid gate
        # multiplies every attention output by ~0.5, compounding to ~0.5^k decay
        # across k recursion levels and suppressing long-range signal
        # (measured on the copy task; see instruction-for-claude/copying-task.md).
        # sigmoid(4) ~= 0.98, and the gate can still learn to close.
        nn.init.constant_(self.gate_linear.bias, 4.0)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        gate = self.gate_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (internally uses float32 for bfloat16 inputs,
        # and enables FlashAttention/memory-efficient kernels when available)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.bool()
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attn_output = attn_output * torch.sigmoid(gate)

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)

        return output

class FFNSwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFNSwiGLU, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x_proj = self.linear1(x)
        x_proj1, x_proj2 = x_proj.chunk(2, dim=-1)
        x_act = torch.nn.functional.silu(x_proj1) * x_proj2
        output = self.linear2(x_act)
        return output

class RecursiveCompressorAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size, compress_size):
        super(RecursiveCompressorAttention, self).__init__()
        self.chunk_size = chunk_size
        self.compress_size = compress_size
        self.register_buffer('mask_tril', torch.ones(chunk_size, chunk_size).tril())
        self.initial_context = nn.Parameter(torch.randn(compress_size, d_model))
        self.norm_mha_encoder = nn.RMSNorm(d_model)
        self.mha_encoder = MultiHeadAttention(d_model, num_heads)
        self.norm_compressor_kv = nn.RMSNorm(d_model)
        self.norm_compressor_q = nn.RMSNorm(d_model)
        self.mha_compressor = MultiHeadAttention(d_model, num_heads)
        self.norm_decompressor_kv = nn.RMSNorm(d_model)
        self.norm_decompressor_q = nn.RMSNorm(d_model)
        self.mha_decompressor = MultiHeadAttention(d_model, num_heads)
        self.compressor_query_pos = nn.Parameter(torch.randn(compress_size, d_model) * 0.02)

    def step(self, xs, hidden):
        """
        xs: list of tensors
            xs[0]: (batch, seq_len, d_model) - main data
            xs[1]: (batch, compressed_seq_len, compress_size, d_model) - compressor query for this level
            xs[2:]: deeper compressor queries
        hidden: list of (inner_context, outer_context) tuples

        Returns: (output_xs, hidden)
            output_xs: list matching xs structure with same shapes
        """
        x = xs[0]
        comp_query = xs[1] if len(xs) >= 2 else None
        deeper_qs = xs[2:]

        batch_size, seq_len, d_model = x.size()

        # Pop current level's hidden state
        if hidden is None:
            hidden = []
        hidden_self = hidden.pop() if hidden else (None, None)
        prev_inner, prev_outer = hidden_self

        # Initial outer context: learnable parameter (data-independent to preserve predict==forward)
        if prev_outer is None:
            prev_outer = self.initial_context[None, :, :].expand(batch_size, -1, -1)

        # Combine with previous partial chunk
        if prev_inner is not None:
            combined = torch.cat([prev_inner, x], dim=1)
            offset = prev_inner.size(1)
        else:
            combined = x
            offset = 0

        total_len = combined.size(1)
        num_full = total_len // self.chunk_size
        rem = total_len % self.chunk_size
        full_len = num_full * self.chunk_size

        # Prepare chunks: full chunks first, then remainder (if any)
        parts = []
        if num_full > 0:
            full_part = combined[:, :full_len].reshape(batch_size * num_full, self.chunk_size, d_model)
            parts.append(full_part)
        if rem > 0:
            rem_part = combined[:, full_len:]
            padding_len = self.chunk_size - rem
            rem_padded = torch.cat([rem_part, torch.zeros(batch_size, padding_len, d_model, dtype=x.dtype, device=x.device)], dim=1)
            parts.append(rem_padded)

        all_chunks = torch.cat(parts, dim=0)

        # Compressor query (per-chunk: (batch, num_full, compress_size, d_model)).
        # First stage gets comp_query=None and derives it from each full chunk's
        # last vector (from `combined`, so step/forward chunk boundaries align).
        # Later stages receive the previous stage's compressed sequence as the
        # query directly.
        if comp_query is None and num_full > 0:
            full_view = combined[:, :full_len].reshape(batch_size, num_full, self.chunk_size, d_model)
            comp_query = full_view[:, :, -1, :].unsqueeze(2).expand(batch_size, num_full, self.compress_size, d_model)
        if comp_query is not None:
            comp_query = comp_query + self.compressor_query_pos

        # Compression / Decompression
        all_pre_norm = all_chunks
        all_normed_for_compressor_kv = self.norm_compressor_kv(all_chunks)
        all_normed_for_decompressor_q = self.norm_decompressor_q(all_chunks)

        comp_query_out = comp_query
        collapsed_dqs = list(deeper_qs)

        if num_full > 0 and comp_query is not None:
            full_normed = all_normed_for_compressor_kv[:batch_size * num_full]

            # comp_query is per-chunk: (batch, num_full, compress_size, d_model).
            assert comp_query.size(1) == num_full, \
                f"comp_query chunk count {comp_query.size(1)} != num_full {num_full}"
            cq_expanded = comp_query.reshape(batch_size * num_full, self.compress_size, d_model)
            cq_expanded_norm = self.norm_compressor_q(cq_expanded)
            compressed = self.mha_compressor(cq_expanded_norm, full_normed, full_normed) + cq_expanded

            # Reshape for recursion: each of compress_size streams processed independently
            compressed = compressed.view(batch_size, num_full, self.compress_size, d_model)
            compressed = compressed.permute(0, 2, 1, 3).contiguous()
            compressed = compressed.view(batch_size * self.compress_size, num_full, d_model)

            # Expand deeper queries for recursive call. Each is per-chunk 4D:
            # (batch, S, compress_size, d_model) -> (batch*compress_size, S, compress_size, d_model)
            expanded_dqs = []
            for dq in deeper_qs:
                if dq is None:
                    expanded_dqs.append(None)
                    continue
                s = dq.size(1)
                exp = dq.unsqueeze(1).expand(batch_size, self.compress_size, s, self.compress_size, d_model)
                exp = exp.reshape(batch_size * self.compress_size, s, self.compress_size, d_model)
                expanded_dqs.append(exp)

            # Recursive step
            recursive_xs = [compressed] + expanded_dqs
            recursive_output, hidden = self.step(recursive_xs, hidden)

            # Extract results
            compressed_out = recursive_output[0]  # (batch*compress_size, num_full, d_model)
            deeper_out = recursive_output[1:]      # list of (batch*compress_size, compress_size, d_model)

            # Reshape compressed back
            compressed_out = compressed_out.view(batch_size, self.compress_size, num_full, d_model)
            compressed_out = compressed_out.permute(0, 2, 1, 3).contiguous()
            # (batch, num_full, compress_size, d_model)

            # Shift: chunk i uses outer context from chunks 0..i-1
            full_outer = torch.cat([prev_outer.unsqueeze(1), compressed_out[:, :-1]], dim=1)
            new_outer = compressed_out[:, -1]

            full_outer = full_outer.view(batch_size * num_full, self.compress_size, d_model)

            if rem > 0:
                all_outer = torch.cat([full_outer, new_outer], dim=0)
            else:
                all_outer = full_outer

            # comp_query output: the full compressed sequence, propagated to the
            # next layer as its per-chunk compressor query (Option B).
            comp_query_out = compressed_out

            # Collapse deeper results across the compress_size stream dimension:
            # (batch*compress_size, S, compress_size, d_model) -> (batch, S, compress_size, d_model)
            collapsed_dqs = []
            for dq_out in deeper_out:
                if dq_out is None:
                    collapsed_dqs.append(None)
                    continue
                s = dq_out.size(1)
                dq_collapsed = dq_out.view(batch_size, self.compress_size, s, self.compress_size, d_model).mean(dim=1)
                collapsed_dqs.append(dq_collapsed)
        else:
            if prev_outer is not None:
                new_outer = prev_outer
                all_outer = prev_outer
            else:
                # No compressor query at all - skip decompression
                new_outer = None
                all_chunks = all_pre_norm
                # Skip decompression block below
                all_outer = None

        if all_outer is not None:
            all_outer_normed = self.norm_decompressor_kv(all_outer)
            all_chunks = self.mha_decompressor(all_normed_for_decompressor_q, all_outer_normed, all_outer_normed)
            all_chunks = all_chunks + all_pre_norm

        # Decoder: causal self-attention + FFN (independent per chunk)
        ac = all_chunks
        all_chunks = self.norm_mha_encoder(all_chunks)
        all_chunks = self.mha_encoder(all_chunks, all_chunks, all_chunks, mask=self.mask_tril)
        all_chunks = all_chunks + ac

        # Reconstruct output
        output_parts = []
        if num_full > 0:
            output_parts.append(all_chunks[:batch_size * num_full].view(batch_size, full_len, d_model))
        if rem > 0:
            rem_start = batch_size * num_full
            output_parts.append(all_chunks[rem_start:rem_start + batch_size, :rem, :])
        total_output = torch.cat(output_parts, dim=1)
        output = total_output[:, offset:offset + seq_len, :]

        # Update hidden state
        new_inner = combined[:, full_len:] if rem > 0 else None
        hidden.append((new_inner, new_outer))

        # Build output list: [processed_data, comp_query_out, *collapsed_deeper_queries].
        # Always include the comp_query_out slot (may be None) so the list length
        # is preserved across layers and recursion levels.
        output_xs = [output, comp_query_out]
        output_xs.extend(collapsed_dqs)

        return output_xs, hidden

    def forward(self, xs):
        output_xs, _ = self.step(xs, None)
        return output_xs

    def predict(self, xs, hidden):
        xs_expanded = [xs[0].unsqueeze(1)] + xs[1:]
        output_xs, hidden = self.step(xs_expanded, hidden)
        output_xs[0] = output_xs[0].squeeze(1)
        return output_xs, hidden

class RecursiveCompressorFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(RecursiveCompressorFFN, self).__init__()
        self.norm = nn.RMSNorm(d_model)
        self.ffn = FFNSwiGLU(d_model, d_ff)

    def forward(self, xs):
        # Pre-norm residual FFN applied per recursion level (each xs element).
        # xs may contain None in unused deeper-query slots (recursion depths
        # beyond where compression bottomed out), so pass those through.
        out = []
        for x in xs:
            if x is None:
                out.append(None)
                continue
            out.append(x + self.ffn(self.norm(x)))
        return out

class RecursiveCompressor(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, chunk_size, compress_size):
        super(RecursiveCompressor, self).__init__()
        self.attention = RecursiveCompressorAttention(d_model, num_heads, chunk_size, compress_size)
        self.ffn = RecursiveCompressorFFN(d_model, d_ff)

    def forward(self, xs):
        xs = self.attention(xs)
        xs = self.ffn(xs)
        return xs
    
    def predict(self, xs, hidden):
        xs, hidden = self.attention.predict(xs, hidden)
        xs = self.ffn(xs)
        return xs, hidden

    def step(self, xs, hidden):
        xs, hidden = self.attention.step(xs, hidden)
        xs = self.ffn(xs)
        return xs, hidden