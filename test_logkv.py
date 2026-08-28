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
    for L in [16, 64, 256, 1024]:
        with torch.no_grad():
            _, (levels, offset) = m.step(torch.randn(1, L, 32), None)
        assert offset == L
        expected_max = math.ceil(math.log(L, 4)) + 1
        assert len(levels) <= expected_max, (L, len(levels), expected_max)
        # each level holds < C current entries and at most one C-sized prev chunk
        for lvl in levels:
            assert lvl[0].size(1) < 4
            if lvl[3] is not None:
                assert lvl[3].size(1) == 4


def test_step_causal_matches_incremental_prefix():
    """step途中のhiddenからの継続が、長い系列のforwardの対応部分と一致"""
    m = make()
    x = torch.randn(1, 50, 32)
    with torch.no_grad():
        y_full = m(x)
        _, h = m.step(x[:, :23], None)
        y_tail, _ = m.step(x[:, 23:], h)
    torch.testing.assert_close(y_tail, y_full[:, 23:], atol=1e-5, rtol=1e-4)


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
# Standalone reference implementation of the LogKV semantics (the original
# parallel forward: build all levels via recursive compression, then gather
# block-aligned slots). forward() now delegates to step(), so equivalence
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
    out = out.view(batch_size, H, seq_len, dh).transpose(1, 2).reshape(batch_size, seq_len, dim)
    return m.lo(out)


def _reference_core(m, q, k, v):
    C = m.chunk_size
    batch_size, seq_len, d_model = q.size()
    x = q

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
        use_prev = c_idx[None, :] >= j[:, None]
        blk = u[:, None] - use_prev.long()
        invalid = blk < 0
        blk = blk.clamp(min=0)
        cexp = c_idx[None, :].expand(seq_len, C)
        logits = torch.einsum('bld,blcd->blc', q, kc[:, blk, cexp, :]) * scale
        if m.level_decay is not None:   # per-head learnable slope (rows are batch-major, head-minor)
            logits = logits - i * m.level_decay.repeat(batch_size // m.num_heads)[:, None, None]
        else:
            logits = logits - i * math.log(C)  # fixed level-decay bias (see LogKV.step)
        logits_list.append(logits.masked_fill(invalid[None, :, :], float('-inf')))
        v_slots_list.append(vc[:, blk, cexp, :])
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


def test_recompute_attention_matches_stored():
    """activation checkpoint経路(訓練時)と非checkpoint経路で出力・勾配が一致"""
    torch.manual_seed(0)
    m = LogKV(dim=32, chunk_size=4, num_heads=4).double()
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
