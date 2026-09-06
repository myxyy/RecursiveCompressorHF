"""Tests for LogKV (logkv.py): step()/forward() equivalence.

Run: uv run pytest test_logkv.py -v
"""

import math

import pytest
import torch

from logkv import LogKV


def make(dim=32, chunk_size=4, seed=0, num_heads=4):
    torch.manual_seed(seed)
    return LogKV(dim=dim, chunk_size=chunk_size, num_heads=num_heads).eval()


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1, 3, 4, 5, 16, 17, 64, 100, 257])
def test_step_full_equals_forward(chunk_size, seq_len, num_heads):
    """一括step(hidden=None)がforwardと一致する"""
    m = make(chunk_size=chunk_size, num_heads=num_heads)
    x = torch.randn(2, seq_len, 32)
    with torch.no_grad():
        y_fwd = m(x)
        y_step, _ = m.step(x, None)
    torch.testing.assert_close(y_step, y_fwd, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("splits", [
    [3, 4], [4, 3], [10, 14], [1, 1, 1, 21], [8, 8, 8], [5, 11, 8], [16, 16],
    [64, 36], [1] * 24,
])
def test_step_split_consistency(splits):
    """任意分割でのstep連結がforwardと一致する"""
    m = make()
    total = sum(splits)
    x = torch.randn(2, total, 32)
    with torch.no_grad():
        y_fwd = m(x)
        hidden = None
        parts = []
        pos = 0
        for n in splits:
            y, hidden = m.step(x[:, pos:pos + n], hidden)
            parts.append(y)
            pos += n
        y_step = torch.cat(parts, dim=1)
    torch.testing.assert_close(y_step, y_fwd, atol=1e-5, rtol=1e-4)


def test_token_by_token_fp64_exact():
    """1トークンずつのstepがfp64でforwardと機械精度一致する
    (kv集合・スロット順がforwardと同一であることの強い検証)"""
    m = make().double()
    x = torch.randn(1, 100, 32, dtype=torch.float64)
    with torch.no_grad():
        y_fwd = m(x)
        hidden = None
        parts = []
        for t in range(100):
            y, hidden = m.step(x[:, t:t + 1], hidden)
            parts.append(y)
        y_step = torch.cat(parts, dim=1)
    assert (y_step - y_fwd).abs().max().item() < 1e-12


def test_hidden_not_mutated():
    """渡したhiddenが破壊されない(同じhiddenからの再実行が同結果)"""
    m = make()
    x1 = torch.randn(2, 10, 32)
    x2 = torch.randn(2, 7, 32)
    with torch.no_grad():
        _, h = m.step(x1, None)
        levels, offset = h
        snapshot = [[t.clone() if torch.is_tensor(t) else t for t in lvl]
                    for lvl in levels]
        ya, _ = m.step(x2, h)
        # h must be unchanged
        assert h[1] == offset
        for lvl, snap in zip(h[0], snapshot):
            for t, s in zip(lvl, snap):
                if torch.is_tensor(t):
                    assert torch.equal(t, s)
                else:
                    assert t is None and s is None
        yb, _ = m.step(x2, h)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)


def test_state_size_logarithmic():
    """hiddenのレベル数がO(log_C L)"""
    m = make(chunk_size=4)
    for L in [15, 16, 17, 63, 64, 65, 255, 256, 257, 1024]:
        with torch.no_grad():
            _, (levels, offset) = m.step(torch.randn(1, L, 32), None)
        assert offset == L
        expected_max = math.ceil(math.log(L, 4)) + 1
        assert len(levels) <= expected_max, (L, len(levels), expected_max)
        # Only the incomplete chunk remains, with no full-segment backing
        # storage retained by a small (or empty) tensor view.
        for i, lvl in enumerate(levels):
            assert len(lvl) == 3
            for t in lvl:
                assert t.size(1) == (L // 4**i) % 4
                assert t.untyped_storage().nbytes() == t.numel() * t.element_size()


def test_step_causal_matches_incremental_prefix():
    """step途中のhiddenからの継続が、長い系列のforwardの対応部分と一致"""
    m = make()
    x = torch.randn(1, 50, 32)
    with torch.no_grad():
        y_full = m(x)
        _, h = m.step(x[:, :23], None)
        y_tail, _ = m.step(x[:, 23:], h)
    torch.testing.assert_close(y_tail, y_full[:, 23:], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("chunk_size", [2, 3, 4])
@pytest.mark.parametrize("segmented", [False, True])
def test_visible_slots_partition_the_past(monkeypatch, chunk_size, segmented):
    """Trace actual gathered slots: every past token occurs exactly once.

    One-hot keys preserve the support of each summary under uniform pooling,
    exposing gaps, duplicated intervals and future leakage independently of
    the reference's slot-index arithmetic.
    """
    C = chunk_size
    L = C**3 + 3
    m = LogKV(dim=L, chunk_size=C).double().eval()
    q = torch.zeros(1, L, L, dtype=torch.float64)
    k = v = torch.eye(L, dtype=torch.float64).unsqueeze(0)
    attend = m._level_attention
    captured = []

    def trace(q, k_ctx, v_ctx, local, invalid, level_bias, scale):
        support = k_ctx[0, local, :] > 0
        captured.append((support & ~invalid[..., None]).sum(dim=1))
        return attend(q, k_ctx, v_ctx, local, invalid, level_bias, scale)

    monkeypatch.setattr(m, "_level_attention", trace)
    # Cross powers of C with both single-token calls and non-aligned chunks.
    cuts = sorted({0, 1, C-1, C, C+1, C**2-1, C**2, C**2+1,
                   C**3-1, C**3, C**3+1, L}) if segmented else [0, L]
    hidden = None
    with torch.no_grad():
        for start, end in zip(cuts, cuts[1:]):
            captured.clear()
            _, hidden = m._attend(q[:, start:end], k[:, start:end], v[:, start:end], hidden)
            counts = torch.stack(captured).sum(dim=0)
            expected = torch.arange(L)[None, :] < torch.arange(start, end)[:, None]
            assert torch.equal(counts, expected.long())


@pytest.mark.parametrize("self_slot", [False, True])
def test_refined_diagram_boundary_values(self_slot):
    """At 4 only [0..3] survives; at 5 it is joined by token 4."""
    m = LogKV(dim=1, chunk_size=4, self_slot=self_slot).double().eval()
    q = k = torch.zeros(1, 18, 1, dtype=torch.float64)
    v = torch.arange(18, dtype=torch.float64).view(1, 18, 1)
    # Explicit diagram intervals, independent of the implementation formula.
    intervals = {0: [], 3: [(0, 1), (1, 2), (2, 3)], 4: [(0, 4)],
                 5: [(0, 4), (4, 5)], 16: [(0, 16)], 17: [(0, 16), (16, 17)]}
    with torch.no_grad():
        out, _ = m._attend(q, k, v, None)
    for s, spans in intervals.items():
        spans = spans + ([(s, s + 1)] if self_slot else [])
        weights = [1 / (end - start) for start, end in spans]
        expected = sum(v[0, start:end, 0].mean().item() * w
                       for (start, end), w in zip(spans, weights)) / sum(weights) if spans else 0.0
        assert out[0, s, 0].item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("self_slot", [False, True])
def test_refined_attention_is_causal(self_slot):
    torch.manual_seed(0)
    m = LogKV(dim=16, chunk_size=4, num_heads=4, phase_emb=True,
              phase_levels=2, gated_attention=True, self_slot=self_slot).double().eval()
    x = torch.randn(1, 70, 16, dtype=torch.float64)
    with torch.no_grad():
        original = m(x)
        for end in [1, 4, 5, 16, 17, 64, 65]:
            changed = x.clone()
            changed[:, end:] = torch.randn_like(changed[:, end:]) * 10
            torch.testing.assert_close(m(changed)[:, :end], original[:, :end], atol=1e-12, rtol=0)


def test_refined_split_backward_matches_full():
    """Partial-state copies must preserve gradients across compression carries."""
    torch.manual_seed(0)
    m = LogKV(dim=16, chunk_size=4, num_heads=4, phase_emb=True,
              phase_levels=2, gated_attention=True, self_slot=True,
              learnable_decay=True).double()
    x = torch.randn(2, 70, 16, dtype=torch.float64, requires_grad=True)
    m(x).square().sum().backward()
    expected = [x.grad.clone()] + [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    x.grad = None
    hidden = None
    parts = []
    cuts = [0, 3, 4, 5, 15, 16, 17, 63, 64, 65, 70]
    for start, end in zip(cuts, cuts[1:]):
        out, hidden = m.step(x[:, start:end], hidden)
        parts.append(out)
    torch.cat(parts, dim=1).square().sum().backward()
    actual = [x.grad] + [p.grad for p in m.parameters()]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("chunk_size", [2, 4])
def test_chunked_step_large_awkward_splits(chunk_size):
    """大きめ系列を境界非整列な塊で処理してもforwardと一致する"""
    m = make(chunk_size=chunk_size)
    x = torch.randn(1, 500, 32)
    with torch.no_grad():
        y_fwd = m(x)
        hidden = None
        parts = []
        pos = 0
        for n in [7, 130, 1, 64, 255, 43]:
            y, hidden = m.step(x[:, pos:pos + n], hidden)
            parts.append(y)
            pos += n
        y_step = torch.cat(parts, dim=1)
    torch.testing.assert_close(y_step, y_fwd, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Standalone reference implementation of the refined LogKV semantics:
# build all levels via recursive compression, then gather only completed
# sub-units in the query's own block. forward() delegates to step(), so equivalence
# tests would be tautological without this independent oracle.
# ---------------------------------------------------------------------------

def reference_forward(m, x):
    """Multi-head wrapper: fold heads into batch, run the single-head
    reference core per head, merge, output-project."""
    batch_size, seq_len, dim = x.size()
    H, dh = m.num_heads, m.head_dim
    if m.phase_emb is not None:
        # base-C digits of absolute position (offset 0), summed over levels
        s = torch.arange(seq_len)
        ph = sum(m.phase_emb[i, (s // (m.chunk_size ** i)) % m.chunk_size]
                 for i in range(m.phase_levels))
        x = x + ph[None].to(x.dtype)

    def split(t):
        return t.view(batch_size, seq_len, H, dh).transpose(1, 2).reshape(batch_size * H, seq_len, dh)

    out = _reference_core(m, split(m.lq(x)), split(m.lk(x)), split(m.lv(x)))
    if m.lg is not None:
        out = out * torch.sigmoid(split(m.lg(x)))
    out = out.view(batch_size, H, seq_len, dh).transpose(1, 2).reshape(batch_size, seq_len, dim)
    return m.lo(out)


def _reference_core(m, q, k, v):
    C = m.chunk_size
    batch_size, seq_len, d_model = q.size()
    x = q
    if m.k_norm is not None:  # kv_norm: level-0 slots
        k = m.k_norm(k)
    if m.v_norm is not None:
        v = m.v_norm(v)

    # build per-level kv chunks by recursive compression
    k_list, v_list = [], []
    ql, kl, vl = q, k, v
    cur_len = seq_len
    while cur_len > C:
        pad = (C - cur_len % C) % C
        if pad > 0:
            z = x.new_zeros(batch_size, pad, d_model)
            ql = torch.cat([ql, z], dim=1)
            kl = torch.cat([kl, z], dim=1)
            vl = torch.cat([vl, z], dim=1)
        n = ql.size(1) // C
        k_list.append(kl.reshape(batch_size, n, C, d_model))
        v_list.append(vl.reshape(batch_size, n, C, d_model))
        q_, k_, v_ = m.compressor(
            ql.reshape(batch_size * n, C, d_model),
            kl.reshape(batch_size * n, C, d_model),
            vl.reshape(batch_size * n, C, d_model))
        if m.k_norm is not None:  # kv_norm: after every compression
            k_ = m.k_norm(k_)
        if m.v_norm is not None:
            v_ = m.v_norm(v_)
        ql = q_.reshape(batch_size, n, d_model)
        kl = k_.reshape(batch_size, n, d_model)
        vl = v_.reshape(batch_size, n, d_model)
        cur_len = n
    # top level: fully-compressed remainder as one padded chunk
    pad = C - cur_len
    if pad > 0:
        z = x.new_zeros(batch_size, pad, d_model)
        kl = torch.cat([kl, z], dim=1)
        vl = torch.cat([vl, z], dim=1)
    k_list.append(kl.reshape(batch_size, 1, C, d_model))
    v_list.append(vl.reshape(batch_size, 1, C, d_model))

    # block-aligned slot gather + one joint softmax
    scale = d_model ** -0.5
    s = torch.arange(seq_len, device=x.device)
    c_idx = torch.arange(C, device=x.device)
    logits_list, v_slots_list = [], []
    for i, (kc, vc) in enumerate(zip(k_list, v_list)):
        sub_len = C ** i
        unit_len = C ** (i + 1)
        u = s // unit_len
        j = (s % unit_len) // sub_len
        invalid = c_idx[None, :] >= j[:, None]
        blk = u[:, None]
        cexp = c_idx[None, :].expand(seq_len, C)
        logits = torch.einsum('bld,blcd->blc', q, kc[:, blk, cexp, :]) * scale
        if m.level_decay is not None:   # per-head learnable slope (rows are batch-major, head-minor)
            logits = logits - i * m.level_decay.repeat(batch_size // m.num_heads)[:, None, None]
        else:
            logits = logits + m.level_sign * i * math.log(C)  # fixed level bias (see LogKV.step)
        logits_list.append(logits.masked_fill(invalid[None, :, :], float('-inf')))
        v_slots_list.append(vc[:, blk, cexp, :])
    if m.self_slot:  # the query's own token as one extra slot (bias 0)
        logits_list.append(torch.einsum('bld,bld->bl', q, k)[..., None] * scale)
        v_slots_list.append(v[:, :, None, :])
    weights = torch.nan_to_num(
        torch.softmax(torch.cat(logits_list, dim=-1), dim=-1))
    all_v = torch.cat(v_slots_list, dim=2)
    return torch.einsum('bls,blsd->bld', weights, all_v)


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1, 3, 4, 5, 16, 17, 64, 100, 257])
def test_forward_matches_reference(chunk_size, seq_len, num_heads):
    """forward(=step委譲)が独立参照実装と一致する"""
    m = make(chunk_size=chunk_size, num_heads=num_heads)
    x = torch.randn(2, seq_len, 32)
    with torch.no_grad():
        y = m(x)
        y_ref = reference_forward(m, x)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("num_heads", [1, 4])
def test_forward_matches_reference_fp64_exact(num_heads):
    """fp64で参照実装と機械精度一致(kv集合・スロット順が厳密同一)"""
    m = make(num_heads=num_heads).double()
    x = torch.randn(1, 257, 32, dtype=torch.float64)
    with torch.no_grad():
        d = (m(x) - reference_forward(m, x)).abs().max().item()
    assert d < 1e-12, d


def test_backward_through_forward():
    """step委譲後のforwardで勾配が全パラメータに流れる"""
    m = make()
    x = torch.randn(2, 100, 32, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# predict (single-token, no sequence dimension)
# ---------------------------------------------------------------------------

def test_predict_shapes():
    m = make()
    y, h = m.predict(torch.randn(2, 32))
    assert y.shape == (2, 32)
    y2, h = m.predict(torch.randn(2, 32), h)
    assert y2.shape == (2, 32) and h[1] == 2


def test_predict_token_by_token_fp64_exact():
    """1トークンずつのpredict連鎖がfp64でforwardと機械精度一致"""
    m = make().double()
    x = torch.randn(1, 100, 32, dtype=torch.float64)
    with torch.no_grad():
        y_fwd = m(x)
        hidden = None
        parts = []
        for t in range(100):
            y, hidden = m.predict(x[:, t], hidden)
            parts.append(y.unsqueeze(1))
        y_pred = torch.cat(parts, dim=1)
    assert (y_pred - y_fwd).abs().max().item() < 1e-12


def test_predict_continues_from_step_prefix():
    """stepで処理したprefixのhiddenからpredictで継続できる"""
    m = make()
    x = torch.randn(2, 60, 32)
    with torch.no_grad():
        y_fwd = m(x)
        _, h = m.step(x[:, :50], None)
        outs = []
        for t in range(50, 60):
            y, h = m.predict(x[:, t], h)
            outs.append(y.unsqueeze(1))
    torch.testing.assert_close(torch.cat(outs, dim=1), y_fwd[:, 50:],
                               atol=1e-5, rtol=1e-4)


def test_heads_are_independent():
    """ヘッドごとに独立した階層・attentionになっている
    (ヘッドhの出力は他ヘッドのk/v射影の摂動に影響されない)"""
    m = make(num_heads=4)
    x = torch.randn(1, 40, 32)
    with torch.no_grad():
        ref = m._attend(m._split_heads(m.lq(x)), m._split_heads(m.lk(x)),
                        m._split_heads(m.lv(x)), None)[0]           # (H, L, dh)
        k = m._split_heads(m.lk(x)).clone()
        v = m._split_heads(m.lv(x)).clone()
        k[1:] += 1.0
        v[1:] += 1.0                                                # perturb heads 1..3
        out = m._attend(m._split_heads(m.lq(x)), k, v, None)[0]
    assert torch.equal(out[0], ref[0])
    assert not torch.allclose(out[1], ref[1])


@pytest.mark.parametrize("self_slot", [False, True])
def test_recompute_attention_matches_stored(self_slot):
    """activation checkpoint経路(訓練時)と非checkpoint経路で出力・勾配が一致"""
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, self_slot=self_slot).double()
    x = torch.randn(2, 100, 32, dtype=torch.float64, requires_grad=True)
    outs, grads = [], []
    for flag in [True, False]:
        m.recompute_attention = flag
        m.zero_grad()
        x.grad = None
        y = m(x)
        y.square().sum().backward()
        outs.append(y.detach().clone())
        grads.append([x.grad.clone()] + [p.grad.clone() for p in m.parameters()])
    assert torch.equal(outs[0], outs[1])
    for g0, g1 in zip(grads[0], grads[1]):
        assert torch.equal(g0, g1)
    m.recompute_attention = True


# ---------------------------------------------------------------------------
# learnable phase embedding
# ---------------------------------------------------------------------------

def make_phase(chunk_size=4, num_heads=4, seed=0):
    torch.manual_seed(seed)
    return LogKV(dim=32, chunk_size=chunk_size, num_heads=num_heads,
                 phase_emb=True, phase_levels=6).eval()


@pytest.mark.parametrize("chunk_size", [2, 4])
def test_phase_forward_matches_reference_fp64(chunk_size):
    m = make_phase(chunk_size=chunk_size).double()
    x = torch.randn(2, 257, 32, dtype=torch.float64)
    with torch.no_grad():
        d = (m(x) - reference_forward(m, x)).abs().max().item()
    assert d < 1e-12, d


def test_phase_step_split_and_predict_fp64():
    """位相はオフセット(絶対位置)から計算されるので分割stepでも一致する"""
    m = make_phase().double()
    x = torch.randn(2, 90, 32, dtype=torch.float64)
    with torch.no_grad():
        y_fwd = m(x)
        y1, h = m.step(x[:, :37])
        y2, h = m.step(x[:, 37:70], h)
        parts = [y1, y2]
        for t in range(70, 90):
            y, h = m.predict(x[:, t], h)
            parts.append(y.unsqueeze(1))
        y_seq = torch.cat(parts, dim=1)
    assert (y_seq - y_fwd).abs().max().item() < 1e-12


def test_phase_breaks_uniform_run_degeneracy():
    """同一トークン連続入力で、位相なしでは一致する位置の出力が位相ありでは区別される"""
    x = torch.randn(1, 1, 32).expand(1, 40, 32).contiguous()  # all positions identical
    torch.manual_seed(0)
    m0 = LogKV(dim=32, chunk_size=4, num_heads=4).eval()
    with torch.no_grad():
        y0 = m0(x)
    # without phase: positions 18/19 (same level phases except digit 0) coincide
    assert torch.allclose(y0[0, 18], y0[0, 19], atol=1e-6)
    m1 = make_phase()
    with torch.no_grad():
        y1 = m1(x)
    assert not torch.allclose(y1[0, 18], y1[0, 19], atol=1e-4)


def test_phase_digits():
    """_phaseが位置のC進数桁に対応するベクトル和になっている"""
    m = make_phase(chunk_size=4)
    with torch.no_grad():
        ph = m._phase(offset=5, seq_len=3, device=torch.device("cpu"))  # positions 5,6,7
    for t, s in enumerate([5, 6, 7]):
        expect = sum(m.phase_emb[i, (s // 4 ** i) % 4] for i in range(6))
        assert torch.allclose(ph[t], expect)


# ---------------------------------------------------------------------------
# learnable level-decay slope
# ---------------------------------------------------------------------------

def test_learnable_decay_init_matches_fixed():
    """初期値log Cなので固定減衰と出力が一致する"""
    torch.manual_seed(0)
    m_fixed = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2).double().eval()
    torch.manual_seed(0)
    m_learn = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
                    learnable_decay=True).double().eval()
    assert torch.allclose(m_learn.level_decay, torch.full((4,), math.log(4), dtype=torch.float64))
    x = torch.randn(2, 100, 32, dtype=torch.float64)
    with torch.no_grad():
        # the parameter is created in fp32 (log C rounded to ~1e-8), hence 1e-6
        assert (m_fixed(x) - m_learn(x)).abs().max().item() < 1e-6


def test_learnable_decay_matches_reference_and_step_fp64():
    """ヘッドごとに異なる係数でも参照実装・分割stepと機械精度一致"""
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, learnable_decay=True).double().eval()
    with torch.no_grad():
        m.level_decay.copy_(torch.tensor([0.0, 0.7, 1.386, 3.0], dtype=torch.float64))
    x = torch.randn(2, 130, 32, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        assert (y - reference_forward(m, x)).abs().max().item() < 1e-12
        y1, h = m.step(x[:, :57]); y2, _ = m.step(x[:, 57:], h)
        assert (torch.cat([y1, y2], 1) - y).abs().max().item() < 1e-12


def test_learnable_decay_gets_gradient():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, learnable_decay=True)
    x = torch.randn(2, 60, 32)
    m(x).square().sum().backward()
    assert m.level_decay.grad is not None and torch.isfinite(m.level_decay.grad).all()
    assert m.level_decay.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# gated attention
# ---------------------------------------------------------------------------

def test_gated_matches_reference_and_step_fp64():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
              gated_attention=True).double().eval()
    with torch.no_grad():
        m.lg.bias.normal_()  # make the gate non-trivial
    x = torch.randn(2, 130, 32, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        assert (y - reference_forward(m, x)).abs().max().item() < 1e-12
        y1, h = m.step(x[:, :57]); y2, _ = m.step(x[:, 57:], h)
        assert (torch.cat([y1, y2], 1) - y).abs().max().item() < 1e-12


def test_gate_changes_output_and_gets_gradient():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, gated_attention=True)
    x = torch.randn(2, 60, 32)
    y = m(x)
    y.square().sum().backward()
    assert m.lg.weight.grad is not None and m.lg.weight.grad.abs().sum() > 0
    with torch.no_grad():
        m.lg.bias.fill_(10.0)  # gate -> ~1
        y_open = m(x)
    assert not torch.allclose(y, y_open)


# ---------------------------------------------------------------------------
# kv_norm
# ---------------------------------------------------------------------------

def test_kv_norm_matches_reference_and_step_fp64():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
              gated_attention=True, kv_norm=True).double().eval()
    with torch.no_grad():
        m.k_norm.weight.normal_(1.0, 0.3); m.v_norm.weight.normal_(1.0, 0.3)
    x = torch.randn(2, 130, 32, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        assert (y - reference_forward(m, x)).abs().max().item() < 1e-12
        y1, h = m.step(x[:, :57]); y2, _ = m.step(x[:, 57:], h)
        assert (torch.cat([y1, y2], 1) - y).abs().max().item() < 1e-12
        for t in range(3):
            y3, h = m.predict(x[:, 57 + t] if False else x[:, t], None if t == 0 else h)


def test_kv_norm_equalizes_slot_scale_across_levels():
    """全レベルのk/vスロットがRMS 1(ゲイン1)になる"""
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, kv_norm=True).eval()
    x = torch.randn(1, 200, 32) * 5.0
    with torch.no_grad():
        _, (levels, _) = m.step(x)
    for i, (cur_q, cur_k, cur_v) in enumerate(levels):
        for t in (cur_k, cur_v):
            if t is not None and t.numel():
                rms = t.pow(2).mean(-1).sqrt()
                assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (i, rms)


def test_kv_norm_gets_gradient():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, kv_norm=True)
    m(torch.randn(2, 60, 32)).square().sum().backward()
    assert m.k_norm.weight.grad.abs().sum() > 0 and m.v_norm.weight.grad.abs().sum() > 0


def test_level_amplify_matches_reference_and_step_fp64():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
              gated_attention=True, kv_norm=True, level_amplify=True).double().eval()
    assert m.level_sign == 1.0
    x = torch.randn(2, 130, 32, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        assert (y - reference_forward(m, x)).abs().max().item() < 1e-12
        y1, h = m.step(x[:, :57]); y2, _ = m.step(x[:, 57:], h)
        assert (torch.cat([y1, y2], 1) - y).abs().max().item() < 1e-12
    # sign actually flips the output relative to decay
    torch.manual_seed(0)
    m2 = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
               gated_attention=True, kv_norm=True).double().eval()
    with torch.no_grad():
        assert not torch.allclose(m2(x), y)


def test_v_norm_only_matches_reference_and_step_fp64():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
              gated_attention=True, v_norm_only=True).double().eval()
    assert m.k_norm is None and m.v_norm is not None
    with torch.no_grad():
        m.v_norm.weight.normal_(1.0, 0.3)
    x = torch.randn(2, 130, 32, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        assert (y - reference_forward(m, x)).abs().max().item() < 1e-12
        y1, h = m.step(x[:, :57]); y2, _ = m.step(x[:, 57:], h)
        assert (torch.cat([y1, y2], 1) - y).abs().max().item() < 1e-12


def test_v_norm_only_normalizes_v_not_k():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, v_norm_only=True).eval()
    x = torch.randn(1, 200, 32) * 5.0
    with torch.no_grad():
        _, (levels, _) = m.step(x)
    v_rms, k_rms = [], []
    for cur_q, cur_k, cur_v in levels:
        for t, acc in [(cur_v, v_rms), (cur_k, k_rms)]:
            if t is not None and t.numel():
                acc.append(t.pow(2).mean(-1).sqrt().flatten())
    v_rms, k_rms = torch.cat(v_rms), torch.cat(k_rms)
    assert torch.allclose(v_rms, torch.ones_like(v_rms), atol=1e-4)
    assert not torch.allclose(k_rms, torch.ones_like(k_rms), atol=0.1)


# ---------------------------------------------------------------------------
# self_slot
# ---------------------------------------------------------------------------

def test_self_slot_matches_reference_step_predict_fp64():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=True, phase_levels=2,
              gated_attention=True, self_slot=True).double().eval()
    x = torch.randn(2, 130, 32, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        assert (y - reference_forward(m, x)).abs().max().item() < 1e-12
        y1, h = m.step(x[:, :57]); y2, h = m.step(x[:, 57:120], h)
        parts = [y1, y2]
        for t in range(120, 130):
            yt, h = m.predict(x[:, t], h); parts.append(yt.unsqueeze(1))
        assert (torch.cat(parts, 1) - y).abs().max().item() < 1e-12


def test_self_slot_position0_attends_to_itself():
    """位置0は従来ゼロ出力だったが、self_slotでは自分のvalueに基づく出力になる"""
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, self_slot=True).eval()
    x = torch.randn(1, 5, 32)
    with torch.no_grad():
        y = m(x)
        v0 = m._split_heads(m.lv(x))[:, 0]                       # (H, dh)
        expect = m.lo(v0.reshape(1, 1, -1))[0, 0]
    assert torch.allclose(y[0, 0], expect, atol=1e-5)
    m0 = LogKV(dim=32, chunk_size=4, num_heads=4).eval()
    with torch.no_grad():
        assert m0(x)[0, 0].abs().max().item() == 0.0


def test_self_slot_changes_output_and_gets_grad():
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4, self_slot=True)
    x = torch.randn(2, 40, 32, requires_grad=True)
    m(x).square().sum().backward()
    assert torch.isfinite(x.grad).all() and m.lk.weight.grad.abs().sum() > 0
