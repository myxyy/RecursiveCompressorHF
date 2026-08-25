import torch
import torch.nn as nn

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
    def __init__(self, dim, chunk_size):
        super(LogKV, self).__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.lq = nn.Linear(dim, dim, bias=False)
        self.lk = nn.Linear(dim, dim, bias=False)
        self.lv = nn.Linear(dim, dim, bias=False)
        self.compressor = Compressor()

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
        last C sub-unit summaries at every scale, so each position sees
        C * num_levels kv entries covering ~chunk_size**num_levels tokens.
        One softmax normalizes across all levels jointly.
        """
        C = self.chunk_size
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.dim, "Input dimension must match the specified dimension"
        q = self.lq(x)  # (batch_size, seq_len, d_model)
        k = self.lk(x)
        v = self.lv(x)

        k_list, v_list = self.forward_list_list(q, k, v)

        scale = d_model ** -0.5
        s = torch.arange(seq_len, device=x.device)
        c_idx = torch.arange(C, device=x.device)

        logits_list = []   # per level: (batch_size, seq_len, C)
        v_slots_list = []  # per level: (batch_size, seq_len, C, d_model)
        for i, (kc, vc) in enumerate(zip(k_list, v_list)):
            # kc, vc: (batch_size, num_blocks_i, C, d_model)
            sub_len = C ** i
            unit_len = C ** (i + 1)
            u = s // unit_len                           # (seq_len,) block index
            j = (s % unit_len) // sub_len               # (seq_len,) sub-unit pos in block
            use_prev = c_idx[None, :] >= j[:, None]     # (seq_len, C) slot taken from block u-1
            blk = u[:, None] - use_prev.long()          # (seq_len, C)
            invalid = blk < 0                           # before sequence start
            blk = blk.clamp(min=0)
            cexp = c_idx[None, :].expand(seq_len, C)
            k_slots = kc[:, blk, cexp, :]               # (batch_size, seq_len, C, d_model)
            v_slots = vc[:, blk, cexp, :]
            logits = torch.einsum('bld,blcd->blc', q, k_slots) * scale
            logits = logits.masked_fill(invalid[None, :, :], float('-inf'))
            logits_list.append(logits)
            v_slots_list.append(v_slots)

        attention_logits = torch.cat(logits_list, dim=-1)        # (batch_size, seq_len, C*levels)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        # Positions with no valid kv at all (sequence start) give an all -inf
        # row -> NaN after softmax; treat them as zero output.
        attention_weights = torch.nan_to_num(attention_weights)
        all_v = torch.cat(v_slots_list, dim=2)                   # (batch_size, seq_len, C*levels, d_model)
        v_out = torch.einsum('bls,blsd->bld', attention_weights, all_v)
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
        prev_k, prev_v] (incomplete current chunk + last completed chunk),
        offset = number of tokens processed so far. The caller's hidden is
        not mutated. Returns (out, new_hidden).
        """
        C = self.chunk_size
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.dim, "Input dimension must match the specified dimension"
        q_new = self.lq(x)
        k_new = self.lk(x)
        v_new = self.lv(x)
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
            return torch.zeros_like(x), (levels, offset + seq_len)

        s_abs = offset + torch.arange(seq_len, device=x.device)  # (seq_len,)
        c_idx = torch.arange(C, device=x.device)
        logits_list, v_slots_list = [], []
        for i, (base, k_ctx, v_ctx) in enumerate(ctxs):
            q_sub = s_abs // (C ** i)                                    # (seq_len,)
            a = q_sub[:, None] - C + ((c_idx[None, :] - q_sub[:, None]) % C)  # (seq_len, C)
            invalid = a < 0
            local = (a - base).clamp(min=0, max=max(k_ctx.size(1) - 1, 0))
            k_slots = k_ctx[:, local, :]                                 # (batch_size, seq_len, C, d_model)
            v_slots = v_ctx[:, local, :]
            logits = torch.einsum('bld,blcd->blc', q_new, k_slots) * scale
            logits = logits.masked_fill(invalid[None, :, :], float('-inf'))
            logits_list.append(logits)
            v_slots_list.append(v_slots)

        attention_logits = torch.cat(logits_list, dim=-1)
        attention_weights = torch.nan_to_num(torch.softmax(attention_logits, dim=-1))
        all_v = torch.cat(v_slots_list, dim=2)
        v_out = torch.einsum('bls,blsd->bld', attention_weights, all_v)
        return v_out, (levels, offset + seq_len)

    def forward_list_list(self, q, k, v):
        chunk_size = self.chunk_size
        batch_size, seq_len, d_model = q.size()
        assert k.size() == (batch_size, seq_len, d_model), "Key tensor shape must match query tensor shape"
        assert v.size() == (batch_size, seq_len, d_model), "Value tensor shape must match query tensor shape"

        k_list = []
        v_list = []

        while seq_len > chunk_size:
            pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
            if pad_len > 0:
                q = torch.cat([q, torch.zeros(batch_size, pad_len, d_model, device=q.device)], dim=1)
                k = torch.cat([k, torch.zeros(batch_size, pad_len, d_model, device=k.device)], dim=1)
                v = torch.cat([v, torch.zeros(batch_size, pad_len, d_model, device=v.device)], dim=1)
            num_chunks = q.size(1) // chunk_size

            k_list.append(k.reshape(batch_size, num_chunks, chunk_size, d_model))
            v_list.append(v.reshape(batch_size, num_chunks, chunk_size, d_model))

            q_, k_, v_ = self.compressor(
                q.reshape(batch_size * num_chunks, chunk_size, d_model),
                k.reshape(batch_size * num_chunks, chunk_size, d_model),
                v.reshape(batch_size * num_chunks, chunk_size, d_model))
            q_ = q_.reshape(batch_size, num_chunks, d_model)
            k_ = k_.reshape(batch_size, num_chunks, d_model)
            v_ = v_.reshape(batch_size, num_chunks, d_model)

            q = q_
            k = k_
            v = v_
            seq_len = q.size(1)

        # Top level: the fully-compressed remainder (length <= chunk_size)
        # carries the sequence's oldest/global information. Include it as a
        # final level, padded to a single chunk, so the receptive field covers
        # the WHOLE sequence rather than only the ~chunk_size**num_levels most
        # recent tokens. (When the input itself is <= chunk_size this is the
        # only level, giving short sequences plain previous-token attention.)
        pad_len = chunk_size - seq_len
        if pad_len > 0:
            k = torch.cat([k, torch.zeros(batch_size, pad_len, d_model, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(batch_size, pad_len, d_model, device=v.device)], dim=1)
        k_list.append(k.reshape(batch_size, 1, chunk_size, d_model))
        v_list.append(v.reshape(batch_size, 1, chunk_size, d_model))

        return k_list, v_list
