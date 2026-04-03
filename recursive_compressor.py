import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
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

class RecursiveCompressor(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, chunk_size, compress_size):
        super(RecursiveCompressor, self).__init__()
        self.chunk_size = chunk_size
        self.compress_size = compress_size
        self.register_buffer('mask_tril', torch.ones(chunk_size, chunk_size).tril())
        self.compressor_query = nn.Parameter(torch.randn(compress_size, d_model))
        self.norm_mha_encoder = nn.LayerNorm(d_model)
        self.mha_encoder = MultiHeadAttention(d_model, num_heads)
        self.norm_ffn_encoder = nn.LayerNorm(d_model)
        self.ffn_encoder = FFNSwiGLU(d_model, d_ff)
        self.norm_mha_decoder = nn.LayerNorm(d_model)
        self.mha_decoder = MultiHeadAttention(d_model, num_heads)
        self.norm_ffn_decoder = nn.LayerNorm(d_model)
        self.ffn_decoder = FFNSwiGLU(d_model, d_ff)
        self.norm_compressor = nn.LayerNorm(d_model)
        self.mha_compressor = MultiHeadAttention(d_model, num_heads)
        self.norm_decompressor = nn.LayerNorm(d_model)
        self.mha_decompressor = MultiHeadAttention(d_model, num_heads)

    def step(self, x, hidden):
        batch_size, seq_len, d_model = x.size()

        # Pop current level's hidden state
        if hidden is None:
            hidden = []
        hidden_self = hidden.pop() if hidden else (None, None)
        prev_inner, prev_outer = hidden_self

        if prev_outer is None:
            prev_outer = self.compressor_query[None, :, :].expand(batch_size, self.compress_size, d_model)

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
        # Kept separate (not interleaved by batch) so we can slice by dim=0 later
        parts = []
        if num_full > 0:
            full_part = combined[:, :full_len].reshape(batch_size * num_full, self.chunk_size, d_model)
            parts.append(full_part)
        if rem > 0:
            rem_part = combined[:, full_len:]
            padding_len = self.chunk_size - rem
            rem_padded = torch.cat([rem_part, torch.zeros(batch_size, padding_len, d_model, device=x.device)], dim=1)
            parts.append(rem_padded)

        all_chunks = torch.cat(parts, dim=0)

        # Encoder: causal self-attention + FFN (independent per chunk)
        ac = all_chunks
        all_chunks = self.norm_mha_encoder(all_chunks)
        all_chunks = self.mha_encoder(all_chunks, all_chunks, all_chunks, mask=self.mask_tril)
        all_chunks = all_chunks + ac

        ac = all_chunks
        all_chunks = self.norm_ffn_encoder(all_chunks)
        all_chunks = self.ffn_encoder(all_chunks)
        all_chunks = all_chunks + ac

        # Compression / Decompression
        all_pre_norm = all_chunks
        all_normed = self.norm_compressor(all_chunks)

        if num_full > 0:
            full_normed = all_normed[:batch_size * num_full]

            comp_query = self.compressor_query[None, :, :].expand(batch_size * num_full, self.compress_size, d_model)
            compressed = self.mha_compressor(comp_query, full_normed, full_normed)

            # Reshape for recursion: each of compress_size streams processed independently
            compressed = compressed.view(batch_size, num_full, self.compress_size, d_model)
            compressed = compressed.permute(0, 2, 1, 3).contiguous()
            compressed = compressed.view(batch_size * self.compress_size, num_full, d_model)

            # Recursive step
            compressed, hidden = self.step(compressed, hidden)

            compressed = compressed.view(batch_size, self.compress_size, num_full, d_model)
            compressed = compressed.permute(0, 2, 1, 3).contiguous()
            # (batch, num_full, compress_size, d_model)

            # Shift: chunk i uses outer context from chunks 0..i-1
            full_outer = torch.cat([prev_outer.unsqueeze(1), compressed[:, :-1]], dim=1)
            new_outer = compressed[:, -1]

            full_outer = full_outer.view(batch_size * num_full, self.compress_size, d_model)

            if rem > 0:
                all_outer = torch.cat([full_outer, new_outer], dim=0)
            else:
                all_outer = full_outer
        else:
            new_outer = prev_outer
            all_outer = prev_outer

        all_outer_normed = self.norm_decompressor(all_outer)
        all_chunks = self.mha_decompressor(all_normed, all_outer_normed, all_outer_normed)
        all_chunks = all_chunks + all_pre_norm

        # Decoder: causal self-attention + FFN (independent per chunk)
        ac = all_chunks
        all_chunks = self.norm_mha_decoder(all_chunks)
        all_chunks = self.mha_decoder(all_chunks, all_chunks, all_chunks, mask=self.mask_tril)
        all_chunks = all_chunks + ac

        ac = all_chunks
        all_chunks = self.norm_ffn_decoder(all_chunks)
        all_chunks = self.ffn_decoder(all_chunks)
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

        return output, hidden

    def forward(self, x):
        output, _ = self.step(x, None)
        return output

    def predict(self, x, hidden):
        output, hidden = self.step(x.unsqueeze(1), hidden)
        return output.squeeze(1), hidden

