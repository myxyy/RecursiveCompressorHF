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
        """Sequential forward with carried state: concatenating step() outputs
        over any split of the input reproduces forward(x) exactly (same kv
        sets, same slot order, so identical up to float reduction order).

        Per-level state: the incomplete current chunk (cur_*, < chunk_size
        entries: the completed sub-unit summaries of the query's own block)
        and the last completed chunk (prev_*): O(chunk_size * num_levels * d)
        memory, i.e. logarithmic in total sequence length. Levels are created
        on demand as chunks cascade upward (the counterpart of forward's
        while-loop depth plus the top-remainder level).

        Why this matches forward(): forward's attention only ever reads slots
        c < j of block u (sub-units entirely before the query) and slots
        c >= j of block u-1 (a fully completed chunk), so padded/incomplete
        chunk compressions are never consumed — exactly the cur/prev state
        kept here, with compression performed only when a chunk completes.

        hidden: list of [cur_q, cur_k, cur_v, prev_k, prev_v] per level
        (None to start). The caller's hidden is not mutated.
        Returns (out, new_hidden).
        """
        C = self.chunk_size
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.dim, "Input dimension must match the specified dimension"
        q_all = self.lq(x)
        k_all = self.lk(x)
        v_all = self.lv(x)
        scale = d_model ** -0.5

        levels = [list(lvl) for lvl in hidden] if hidden is not None else []

        def new_level():
            e = x.new_zeros(batch_size, 0, d_model)
            return [e, e, e, None, None]

        def push(idx, q1, k1, v1):
            if idx == len(levels):
                levels.append(new_level())
            lvl = levels[idx]
            lvl[0] = torch.cat([lvl[0], q1], dim=1)
            lvl[1] = torch.cat([lvl[1], k1], dim=1)
            lvl[2] = torch.cat([lvl[2], v1], dim=1)
            if lvl[0].size(1) == C:
                # chunk complete: it becomes this level's previous block and
                # its compression cascades to the level above
                lvl[3], lvl[4] = lvl[1], lvl[2]
                q_, k_, v_ = self.compressor(lvl[0], lvl[1], lvl[2])
                e = x.new_zeros(batch_size, 0, d_model)
                lvl[0], lvl[1], lvl[2] = e, e, e
                push(idx + 1, q_, k_, v_)

        outs = []
        for t in range(seq_len):
            q_t = q_all[:, t:t + 1]  # (batch_size, 1, d_model)

            # Attention over the current window state (token t not yet pushed,
            # matching forward where slot c == j falls through to prev).
            # Slot order per level: cur[0..j-1] then prev[j..C-1] — identical
            # to forward's slot order c = 0..C-1 with invalid slots removed.
            k_slots, v_slots = [], []
            for lvl in levels:
                cur_k, cur_v, prev_k, prev_v = lvl[1], lvl[2], lvl[3], lvl[4]
                j = cur_k.size(1)
                if prev_k is not None:
                    k_slots.append(torch.cat([cur_k, prev_k[:, j:]], dim=1))
                    v_slots.append(torch.cat([cur_v, prev_v[:, j:]], dim=1))
                elif j > 0:
                    k_slots.append(cur_k)
                    v_slots.append(cur_v)
            if k_slots:
                ks = torch.cat(k_slots, dim=1)  # (batch_size, n_slots, d_model)
                vs = torch.cat(v_slots, dim=1)
                logits = torch.einsum('bod,bsd->bos', q_t, ks) * scale
                weights = torch.softmax(logits, dim=-1)
                outs.append(torch.einsum('bos,bsd->bod', weights, vs))
            else:
                outs.append(torch.zeros_like(q_t))  # sequence start: no kv yet

            push(0, q_t, k_all[:, t:t + 1], v_all[:, t:t + 1])

        return torch.cat(outs, dim=1), levels

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
