"""Grouped QK/PV statistics and gradients against the streaming gather path."""
from unittest.mock import patch

import pytest
import torch

from models.logkv.attention import LogKV


@pytest.mark.parametrize('chunk', [2, 3, 4])
@pytest.mark.parametrize('level', [0, 1, 2])
@pytest.mark.parametrize('case', ['all_masked', 'partial', 'aligned', 'full_and_tail'])
def test_grouped_level_statistics_and_gradients(chunk, level, case):
    torch.manual_seed(92)
    sub_len = chunk ** level
    length = {'all_masked': sub_len, 'partial': sub_len + 1,
              'aligned': chunk * sub_len, 'full_and_tail': 2 * chunk * sub_len - 1}[case]
    module = LogKV(8, chunk, num_heads=2).double()
    q = torch.randn(4, length, 4, dtype=torch.double, requires_grad=True)
    k = torch.randn(4, length // sub_len, 4, dtype=torch.double, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    # Different per-head slopes, including amplification; preserve bias grads.
    bias = torch.tensor([-1., .5, -1., .5], dtype=torch.double).reshape(4, 1, 1).requires_grad_()
    q_sub = torch.arange(length) // sub_len
    j = q_sub % chunk
    slots = torch.arange(chunk - 1)
    local = ((q_sub - j)[:, None] + slots).clamp(max=k.size(1) - 1)
    invalid = slots[None] >= j[:, None]
    expected = module._level_attention(q, k, v, local, invalid, bias, .5)
    actual = module._grouped_level_attention(q, k, v, sub_len, bias, .5)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def objective(stats):
        m, l, acc = stats
        return torch.where(torch.isfinite(m), m, 0).sum() + l.square().sum() + acc.square().sum()

    ga = torch.autograd.grad(objective(actual), (q, k, v, bias))
    ge = torch.autograd.grad(objective(expected), (q, k, v, bias))
    for a, b in zip(ga, ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('self_slot', [False, True])
def test_full_forward_and_backward_do_not_gather_slots(recompute, self_slot):
    model = LogKV(8, 4, num_heads=2, self_slot=self_slot, gated_attention=True).double()
    model.recompute_attention = recompute
    x = torch.randn(2, 67, 8, dtype=torch.double, requires_grad=True)
    reference, _ = model.step(x)
    with patch.object(model, '_attend_contexts', side_effect=AssertionError('streaming indices')), \
         patch.object(model, '_level_attention', side_effect=AssertionError('per-query K/V gather')):
        actual = model(x)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('self_slot', [False, True])
def test_empty_full_forward(self_slot):
    model = LogKV(8, 4, num_heads=2, self_slot=self_slot)
    x = torch.empty(2, 0, 8)
    torch.testing.assert_close(model(x), model.step(x)[0])
