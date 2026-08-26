import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Compressor(nn.Module):
    def __init__(self):
        super(Compressor, self).__init__()

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.size()
        assert k.size() == (batch_size, seq_len, d_model), "Key tensor shape must match query tensor shape"
        assert v.size() == (batch_size, seq_len, d_model), "Value tensor shape must match query tensor shape"

        q_out = q[:, -1, :].unsqueeze(1)  # Take the last query vector and keep the batch dimension
        attention_logits = torch.bmm(q_out, k.transpose(1, 2)) * (d_model ** -0.5)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        k_out = torch.bmm(attention_weights, k)
        v_out = torch.bmm(attention_weights, v)

        return q_out, k_out, v_out

class LogKV(nn.Module):
    def __init__(self, dim, chunk_size, num_heads=1):
        super(LogKV, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lq = nn.Linear(dim, dim, bias=False)
        self.lk = nn.Linear(dim, dim, bias=False)
        self.lv = nn.Linear(dim, dim, bias=False)
        self.lo = nn.Linear(dim, dim, bias=False)
        self.compressor = Compressor()
        # Recompute each level's slot gather + partial softmax in backward
        # (activation checkpointing) instead of storing the gathered
        # (B*H, L, C, head_dim) slot tensors for every level — the dominant
        # VRAM term. Numerically identical to storing them; costs one extra
        # gather/einsum per level in backward. Inference is unaffected.
        self.recompute_attention = True

    def _split_heads(self, t):
        """(B, L, dim) -> (B*H, L, head_dim): heads folded into the batch
        dimension so the hierarchy/attention core runs per head unchanged."""
        batch_size, seq_len, _ = t.size()
        t = t.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return t.transpose(1, 2).reshape(batch_size * self.num_heads, seq_len, self.head_dim)

    def _merge_heads(self, t, batch_size):
        """(B*H, L, head_dim) -> (B, L, dim)"""
        seq_len = t.size(1)
        t = t.view(batch_size, self.num_heads, seq_len, self.head_dim)
        return t.transpose(1, 2).reshape(batch_size, seq_len, self.dim)

    def forward(self, x):
        """Multi-resolution causal attention over log-many kv slots.

        Semantics (per level i, sub-unit = chunk_size**i tokens, block =
        chunk_size sub-units): a query at position s, sitting in sub-unit
        j (0..C-1) of block u, attends to C slots — the compressed kv of the
        C sub-units immediately PRECEDING its own sub-unit. Slots c < j come
        from the query's own block u (already-complete sub-units); slots
        c >= j come from the previous block u-1 (their own-block versions are
        incomplete or would contain the query's local future, so using them
        would break causality). Together they form a sliding window of the
        last C sub-unit summaries at every scale; the top level (the fully
        compressed remainder) extends the receptive field to the whole
        sequence. One softmax normalizes across all levels jointly.

        step() is the single authoritative implementation (its chunked pass
        reproduces these semantics exactly — see its docstring); forward is a
        one-shot step() from empty state. A standalone reference
        implementation of the semantics above lives in test_logkv.py.
        """
        v_out, _ = self.step(x)
        return v_out

    def step(self, x, hidden=None):
        """Chunked sequential forward: processes an arbitrary-length segment
        in one vectorized pass (no per-token Python loop; only an
        O(log total_len) loop over levels). Concatenating step() outputs over
        any split of the input reproduces forward(x).

        Key facts making this possible:
        * The block-aligned slot selection (c < j from own block, c >= j from
          the previous block) is exactly the C sub-units immediately
          preceding the query's sub-unit — a plain sliding window — with the
          slot order given in closed form by
              a_c = q_sub - C + ((c - q_sub) mod C).
        * Every window element ends strictly before its query (element
          a <= q_sub-1 completes at token (a+1)*sub_len - 1 < s), so the
          whole hierarchy can be updated FIRST and attention computed
          afterwards for all queries at once without breaking causality.
        * The oldest window position of the segment's first query is
          >= the absolute start of [prev, cur], so per level the context
          [prev, cur, entries completed this call] covers every access.

        hidden: (levels, offset) — levels[i] = [cur_q, cur_k, cur_v,
        prev_k, prev_v] (incomplete current chunk + last completed chunk;
        heads are folded into the batch dimension, i.e. batch B*H),
        offset = number of tokens processed so far. The caller's hidden is
        not mutated. Returns (out, new_hidden).
        """
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.dim, "Input dimension must match the specified dimension"
        q_new = self._split_heads(self.lq(x))
        k_new = self._split_heads(self.lk(x))
        v_new = self._split_heads(self.lv(x))
        v_out, hidden = self._attend(q_new, k_new, v_new, hidden)
        return self.lo(self._merge_heads(v_out, batch_size)), hidden

    def _attend(self, q_new, k_new, v_new, hidden):
        """Hierarchy update + windowed attention on projected, head-folded
        (B*H, L, head_dim) tensors. See step() for the semantics."""
        C = self.chunk_size
        batch_size, seq_len, d_model = q_new.size()
        x = q_new  # for dtype/device
        scale = d_model ** -0.5

        if hidden is None:
            levels, offset = [], 0
        else:
            levels, offset = hidden
            levels = [list(lvl) for lvl in levels]

        # ---- (A) update the hierarchy, recording per-level attention
        #      contexts (base absolute index, [prev | cur | new] kv) ----
        ctxs = []  # per level: (base_abs, k_ctx, v_ctx)
        empty = x.new_zeros(batch_size, 0, d_model)
        nq, nk, nv = q_new, k_new, v_new  # entries arriving at level i
        i = 0
        while nq.size(1) > 0:
            if i == len(levels):
                levels.append([empty, empty, empty, None, None])
            lvl = levels[i]
            cur_q, cur_k, cur_v, prev_k, prev_v = lvl
            m = cur_q.size(1)
            base = offset // (C ** i) - m - (C if prev_k is not None else 0)
            k_parts = ([prev_k] if prev_k is not None else []) + [cur_k, nk]
            v_parts = ([prev_v] if prev_v is not None else []) + [cur_v, nv]
            ctxs.append((base, torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)))

            all_q = torch.cat([cur_q, nq], dim=1)
            all_k = torch.cat([cur_k, nk], dim=1)
            all_v = torch.cat([cur_v, nv], dim=1)
            n_chunks = all_q.size(1) // C
            if n_chunks > 0:
                comp_len = n_chunks * C
                q_, k_, v_ = self.compressor(
                    all_q[:, :comp_len].reshape(batch_size * n_chunks, C, d_model),
                    all_k[:, :comp_len].reshape(batch_size * n_chunks, C, d_model),
                    all_v[:, :comp_len].reshape(batch_size * n_chunks, C, d_model))
                # last completed chunk becomes this level's previous block
                lvl[3] = all_k[:, comp_len - C:comp_len]
                lvl[4] = all_v[:, comp_len - C:comp_len]
                lvl[0] = all_q[:, comp_len:]
                lvl[1] = all_k[:, comp_len:]
                lvl[2] = all_v[:, comp_len:]
                nq = q_.reshape(batch_size, n_chunks, d_model)
                nk = k_.reshape(batch_size, n_chunks, d_model)
                nv = v_.reshape(batch_size, n_chunks, d_model)
            else:
                lvl[0], lvl[1], lvl[2] = all_q, all_k, all_v
                nq = nk = nv = empty
            i += 1
        # levels above the cascade keep their state but still serve attention
        for i2 in range(i, len(levels)):
            cur_q, cur_k, cur_v, prev_k, prev_v = levels[i2]
            m = cur_q.size(1)
            base = offset // (C ** i2) - m - (C if prev_k is not None else 0)
            k_parts = ([prev_k] if prev_k is not None else []) + [cur_k]
            v_parts = ([prev_v] if prev_v is not None else []) + [cur_v]
            ctxs.append((base, torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)))

        # ---- (B) attention for all queries of the segment at once ----
        if not ctxs:
            return torch.zeros_like(q_new), (levels, offset + seq_len)

        s_abs = offset + torch.arange(seq_len, device=x.device)  # (seq_len,)
        c_idx = torch.arange(C, device=x.device)
        k_ctxs, v_ctxs, locals_, invalids = [], [], [], []
        for i, (base, k_ctx, v_ctx) in enumerate(ctxs):
            q_sub = s_abs // (C ** i)                                    # (seq_len,)
            a = q_sub[:, None] - C + ((c_idx[None, :] - q_sub[:, None]) % C)  # (seq_len, C)
            k_ctxs.append(k_ctx)
            v_ctxs.append(v_ctx)
            locals_.append((a - base).clamp(min=0, max=max(k_ctx.size(1) - 1, 0)))
            invalids.append(a < 0)
        n_levels = len(ctxs)
        flat = (*k_ctxs, *v_ctxs, *locals_, *invalids)
        # The whole attention pass is one checkpoint region: autograd keeps
        # only its inputs (per-level contexts, ~1.33*L*d total) and output,
        # and recomputes gathers / partial softmaxes in backward.
        if self.recompute_attention and torch.is_grad_enabled() and q_new.requires_grad:
            v_out = checkpoint(self._attend_levels, q_new, n_levels, scale, *flat,
                               use_reentrant=False)
        else:
            v_out = self._attend_levels(q_new, n_levels, scale, *flat)
        return v_out, (levels, offset + seq_len)

    def _attend_levels(self, q, n_levels, scale, *flat):
        """Online softmax over levels: each level contributes its slot-local
        max m_i, denominator l_i and value numerator acc_i (relative to
        m_i); these are merged with the standard rescaling so the result
        equals one joint softmax over the concatenated slots of all levels —
        without ever materializing the (B*H, L, C*levels, head_dim)
        concatenation."""
        C = self.chunk_size
        k_ctxs, v_ctxs = flat[:n_levels], flat[n_levels:2 * n_levels]
        locals_, invalids = flat[2 * n_levels:3 * n_levels], flat[3 * n_levels:]
        m = l = acc = None
        for i in range(n_levels):
            # Level-decay bias: a level-i slot summarizes C**i tokens, and a
            # salient token survives attention-pooled compression undiluted,
            # so it appears in ~log(L) slots across levels at full strength.
            # Weighting level i by C**-i collapses those cross-level copies
            # into a geometric series ~= one count, removing the
            # multiplicity amplification behind topic fixation (and inducing
            # a parameter-free ~1/distance recency prior).
            level_bias = -i * math.log(C)
            m_i, l_i, acc_i = self._level_attention(
                q, k_ctxs[i], v_ctxs[i], locals_[i], invalids[i], level_bias, scale)
            if m is None:
                m, l, acc = m_i, l_i, acc_i
            else:
                m_new = torch.maximum(m, m_i)
                # rows with no valid slot so far have m_new = -inf; any finite
                # reference works for them since their contributions are 0
                m_ref = torch.where(m_new == float('-inf'), torch.zeros_like(m_new), m_new)
                alpha = torch.exp(m - m_ref)
                beta = torch.exp(m_i - m_ref)
                l = l * alpha + l_i * beta
                acc = acc * alpha[..., None] + acc_i * beta[..., None]
                m = m_new
        # positions with no valid kv at all (sequence start) -> zero output
        l_safe = torch.where(l > 0, l, torch.ones_like(l))
        return (acc / l_safe[..., None]).to(q.dtype)

    @staticmethod
    def _level_attention(q, k_ctx, v_ctx, local, invalid, level_bias, scale):
        """One level's slot gather + partial softmax statistics.
        Returns (m, l, acc): per-row max logit (-inf if no valid slot),
        sum of exp(logit - m), and sum of exp(logit - m) * v."""
        k_slots = k_ctx[:, local, :]                                 # (B*H, L, C, head_dim)
        v_slots = v_ctx[:, local, :]
        logits = torch.einsum('bld,blcd->blc', q, k_slots) * scale + level_bias
        logits = logits.masked_fill(invalid[None, :, :], float('-inf'))
        # accumulate softmax statistics in >= fp32 (fp32 under bf16 autocast,
        # fp64 for fp64 inputs)
        acc_dtype = torch.promote_types(logits.dtype, torch.float32)
        m = logits.max(dim=-1).values.to(acc_dtype)                  # (B*H, L)
        m_safe = torch.where(m == float('-inf'), torch.zeros_like(m), m)
        p = torch.exp(logits.to(acc_dtype) - m_safe[..., None])
        l = p.sum(dim=-1)
        acc = torch.einsum('blc,blcd->bld', p, v_slots).to(acc_dtype)
        return m, l, acc

    def predict(self, x, hidden=None):
        """Single-token inference: x is (batch_size, d_model), one token
        without the sequence dimension (same interface as
        RecursiveCompressorAttention.predict). Returns (out, new_hidden)
        with out of shape (batch_size, d_model)."""
        v_out, hidden = self.step(x.unsqueeze(1), hidden)
        return v_out.squeeze(1), hidden


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


class LogKVBlock(nn.Module):
    """Standard pre-norm transformer block with LogKV attention:
    x = x + LogKV(RMSNorm(x)); x = x + FFNSwiGLU(RMSNorm(x))."""

    def __init__(self, dim, chunk_size, d_ff, num_heads=1):
        super(LogKVBlock, self).__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = LogKV(dim, chunk_size, num_heads)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = FFNSwiGLU(dim, d_ff)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    def step(self, x, hidden=None):
        y, hidden = self.attention.step(self.attention_norm(x), hidden)
        x = x + y
        return x + self.ffn(self.ffn_norm(x)), hidden

    def predict(self, x, hidden=None):
        y, hidden = self.attention.predict(self.attention_norm(x), hidden)
        x = x + y
        return x + self.ffn(self.ffn_norm(x)), hidden
