"""Tests for LogKV (recursive_compressor_2): step()/forward() equivalence.

Run: uv run pytest test_logkv.py -v
"""

import math

import pytest
import torch

from recursive_compressor_2 import LogKV


def make(dim=32, chunk_size=4, seed=0):
    torch.manual_seed(seed)
    return LogKV(dim=dim, chunk_size=chunk_size).eval()


@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1, 3, 4, 5, 16, 17, 64, 100, 257])
def test_step_full_equals_forward(chunk_size, seq_len):
    """一括step(hidden=None)がforwardと一致する"""
    m = make(chunk_size=chunk_size)
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
