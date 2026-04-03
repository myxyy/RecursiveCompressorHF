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

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        original_seq_len = seq_len
        if seq_len % self.chunk_size != 0:
            padding_len = self.chunk_size - (seq_len % self.chunk_size)
            x = torch.cat([x, torch.zeros(batch_size, padding_len, d_model, device=x.device)], dim=1)
            seq_len += padding_len

        x = x.view(batch_size * (seq_len // self.chunk_size), self.chunk_size, d_model)

        x_ = x
        x = self.norm_mha_encoder(x)
        x = self.mha_encoder(x, x, x, mask=self.mask_tril)
        x = x + x_

        x_ = x
        x = self.norm_ffn_encoder(x)
        x = self.ffn_encoder(x)
        x = x + x_

        x_ = x
        if seq_len // self.chunk_size > 1:
            x = self.norm_compressor(x)
            compressor_query = self.compressor_query.unsqueeze(0).expand(batch_size * (seq_len // self.chunk_size), self.compress_size, d_model)
            compressed = self.mha_compressor(compressor_query, x, x)
            compressed = compressed.view(batch_size, seq_len // self.chunk_size, self.compress_size, d_model).permute(0, 2, 1, 3).contiguous().view(batch_size * self.compress_size, seq_len // self.chunk_size, d_model)
            compressed = self.forward(compressed)
            compressed = compressed.view(batch_size, self.compress_size, seq_len // self.chunk_size, d_model).permute(0, 2, 1, 3).contiguous()
            compressed = torch.cat([self.compressor_query[None, None, :, :].expand(batch_size, 1, self.compress_size, d_model), compressed[:, :-1, :, :]], dim=1)
            compressed = compressed.view(batch_size * (seq_len // self.chunk_size), self.compress_size, d_model)
        else:
            compressed = self.compressor_query[None, :, :].expand(batch_size, self.compress_size, d_model)
        compressed = self.norm_decompressor(compressed)
        x = self.mha_decompressor(x, compressed, compressed)
        x = x + x_

        x_ = x
        x = self.norm_mha_decoder(x)
        x = self.mha_decoder(x, x, x, mask=self.mask_tril)
        x = x + x_

        x_ = x
        x = self.norm_ffn_decoder(x)
        x = self.ffn_decoder(x)
        x = x + x_

        return x.view(batch_size, seq_len, d_model)[:, :original_seq_len, :]

    def predict(self, x, hidden: list[tuple[None | torch.Tensor, None | torch.Tensor]] | None):
        batch_size, d_model = x.size()
        if hidden is None:
            hidden = []
        hidden_self = hidden.pop() if hidden else (None, None)
        inner_context, outer_context = hidden_self

        if inner_context is None:
            inner_context = x.unsqueeze(1)
        else:
            inner_context = torch.cat([inner_context, x.unsqueeze(1)], dim=1)

        assert inner_context.size(1) <= self.chunk_size, "Chunk size exceeded in predict"
        inner_context_original = inner_context
        inner_context_length = inner_context.size(1)
        if inner_context.size(1) < self.chunk_size:
            padding_len = self.chunk_size - inner_context.size(1)
            inner_context = torch.cat([inner_context, torch.zeros(batch_size, padding_len, d_model, device=x.device)], dim=1)

        if outer_context is None:
            outer_context = self.compressor_query.unsqueeze(0).expand(batch_size, self.compress_size, d_model)

        inner_context_ = inner_context
        inner_context = self.norm_mha_encoder(inner_context)
        inner_context = self.mha_encoder(inner_context, inner_context, inner_context, mask=self.mask_tril)
        inner_context = inner_context + inner_context_

        inner_context_ = inner_context
        inner_context = self.norm_ffn_encoder(inner_context)
        inner_context = self.ffn_encoder(inner_context)
        inner_context = inner_context + inner_context_

        inner_context = self.norm_compressor(inner_context)
        if inner_context_length == self.chunk_size:
            compressor_query = self.compressor_query.unsqueeze(0).expand(batch_size, self.compress_size, d_model)
            compressed = self.mha_compressor(compressor_query, inner_context, inner_context)
            compressed = compressed.view(batch_size * self.compress_size, d_model)
            compressed, hidden = self.predict(compressed, hidden)
            outer_context = compressed.view(batch_size, self.compress_size, d_model)
        compressed = self.norm_decompressor(outer_context)
        inner_context = self.mha_decompressor(inner_context, compressed, compressed)
        inner_context = inner_context + inner_context_

        inner_context_ = inner_context
        inner_context = self.norm_mha_decoder(inner_context)
        inner_context = self.mha_decoder(inner_context, inner_context, inner_context, mask=self.mask_tril)
        inner_context = inner_context + inner_context_

        inner_context_ = inner_context
        inner_context = self.norm_ffn_decoder(inner_context)
        inner_context = self.ffn_decoder(inner_context)
        inner_context = inner_context + inner_context_

        x = inner_context[:, inner_context_length - 1, :]
        if inner_context_length == self.chunk_size:
            inner_context = None
        else:
            inner_context = inner_context_original
        hidden_self = (inner_context, outer_context)
        if hidden is None:
            hidden = []
        hidden.append(hidden_self)
        return x, hidden

