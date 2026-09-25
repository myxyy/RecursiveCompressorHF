"""Dedicated decode: independent oracle, carries, state reuse and precision."""
import copy
from unittest.mock import patch

import pytest
import torch

from models.logkv.configuration import LogKVConfig
from models.logkv.attention import CausalConvBlock, LogKV, LogKVBlock
from models.logkv.modeling import LogKVLM
from tests.logkv.test_logkv import reference_forward


def assert_state(a, b, *, atol=1e-12, rtol=1e-12):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state(x, y, atol=atol, rtol=rtol)
    else:
        assert a == b


@pytest.mark.parametrize('chunk_size', [2, 3, 4])
@pytest.mark.parametrize('num_heads', [1, 4])
@pytest.mark.parametrize('options', [
    {}, {'self_slot': True}, {'level_amplify': True},
    {'kv_norm': True, 'self_slot': True},
    {'v_norm_only': True, 'gated_attention': True},
    {'phase_emb': True, 'phase_levels': 3, 'learnable_decay': True,
     'gated_attention': True, 'self_slot': True},
])
def test_predict_reference_carries_and_bidirectional_state(chunk_size, num_heads, options):
    torch.manual_seed(42)
    model = LogKV(16, chunk_size, num_heads=num_heads, **options).double().eval()
    if model.level_decay is not None:
        with torch.no_grad():
            model.level_decay.copy_(torch.tensor([-0.7, 0., 0.5, 2.])[:num_heads])
    x = torch.randn(2, 137, 16, dtype=torch.double)
    with torch.no_grad():
        ref = reference_forward(model, x)
        hidden = None
        parts = []
        for token in x.unbind(1):
            out, hidden = model.predict(token, hidden)
            parts.append(out[:, None])
            offset = hidden[1]
            for i, level in enumerate(hidden[0]):
                for value in level:
                    assert value.size(1) == (offset // chunk_size**i) % chunk_size
                    assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        torch.testing.assert_close(torch.cat(parts, 1), ref, atol=1e-12, rtol=1e-12)
        # prefill -> predict across multiple carries -> step again.
        _, hidden = model.step(x[:, :chunk_size**3 - 1])
        saved = copy.deepcopy(hidden)
        parts = []
        for pos in range(chunk_size**3 - 1, chunk_size**3 + 2):
            out, nxt = model.predict(x[:, pos], hidden)
            expected, expected_state = model.step(x[:, pos:pos+1], hidden)
            torch.testing.assert_close(out, expected[:, 0], atol=1e-12, rtol=1e-12)
            assert_state(nxt, expected_state)
            if not parts:
                assert_state(hidden, saved, atol=0, rtol=0)
            hidden = nxt
            parts.append(out)
        tail, _ = model.step(x[:, chunk_size**3+2:], hidden)
        torch.testing.assert_close(tail, ref[:, chunk_size**3+2:], atol=1e-12, rtol=1e-12)


def test_predict_preserves_gradients_across_carries():
    torch.manual_seed(43)
    model = LogKVBlock(8, 3, 16, num_heads=2, phase_emb=True, phase_levels=2,
                       learnable_decay=True, kv_norm=True, self_slot=True,
                       gated_attention=True, conv_kernel_size=4).double()
    x = torch.randn(2, 31, 8, dtype=torch.double, requires_grad=True)
    model(x).square().sum().backward()
    expected = [x.grad.clone()] + [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    x.grad = None
    parts, hidden = [], None
    for token in x.unbind(1):
        out, hidden = model.predict(token, hidden)
        parts.append(out[:, None])
    torch.cat(parts, 1).square().sum().backward()
    for actual, ref in zip([x.grad] + [p.grad for p in model.parameters()], expected):
        torch.testing.assert_close(actual, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('kernel', [1, 4, 7])
def test_conv_predict_window_and_cache(kernel):
    torch.manual_seed(44)
    conv = CausalConvBlock(8, kernel).double()
    x = torch.randn(2, 19, 8, dtype=torch.double)
    with torch.no_grad():
        expected = conv(x)
        cache = None
        for i, token in enumerate(x.unbind(1)):
            saved = None if cache is None else cache.clone()
            out, nxt = conv.predict(token, cache)
            if cache is not None:
                assert torch.equal(cache, saved)
            assert nxt.untyped_storage().nbytes() == nxt.numel() * nxt.element_size()
            torch.testing.assert_close(out, expected[:, i], atol=1e-12, rtol=1e-12)
            cache = nxt


def test_lm_predict_and_hf_decode_do_not_call_step():
    torch.manual_seed(45)
    model = LogKVLM(LogKVConfig(vocab_size=24, d_model=16, d_ff=32, num_heads=4,
        num_layers=2, self_slot=True, gated_attention=True, conv_kernel_size=4,
        bos_token_id=None, eos_token_id=None, pad_token_id=None)).double().eval()
    ids = torch.randint(24, (2, 19))
    with torch.no_grad():
        ref = model(ids).logits
        _, hidden = model.step(ids[:, :15])
        snapshot = copy.deepcopy(hidden)
        with patch.object(LogKVLM, 'step', side_effect=AssertionError('LM.step called')), \
             patch.object(LogKVBlock, 'step', side_effect=AssertionError('Block.step called')), \
             patch.object(LogKV, 'step', side_effect=AssertionError('LogKV.step called')), \
             patch.object(CausalConvBlock, 'step', side_effect=AssertionError('Conv.step called')):
            out, nxt = model.predict(ids[:, 15], hidden)
            torch.testing.assert_close(out, ref[:, 15], atol=1e-12, rtol=1e-12)
            assert_state(hidden, snapshot, atol=0, rtol=0)
            out = model(ids[:, 16:17], past_key_values=nxt, use_cache=True)
            torch.testing.assert_close(out.logits, ref[:, 16:17], atol=1e-12, rtol=1e-12)
            assert out.past_key_values is not None
        with patch.object(model, 'predict', wraps=model.predict) as predict:
            model.generate(ids[:1, :15], max_new_tokens=4, do_sample=False)
            assert predict.call_count == 3
    # Gradient-enabled forward stays on the unchanged training path.
    with patch.object(model, 'predict', side_effect=AssertionError('training dispatched to predict')):
        model(ids[:, :1], past_key_values=hidden).logits.sum().backward()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('precision', ['float32', 'bfloat16', 'autocast'])
@pytest.mark.parametrize('learnable', [False, True])
def test_predict_precision_and_state(device, precision, learnable):
    torch.manual_seed(46)
    model = LogKV(32, 4, num_heads=4, self_slot=True, gated_attention=True,
                  learnable_decay=learnable).eval().to(device)
    if learnable:
        with torch.no_grad():
            model.level_decay.copy_(torch.tensor([-0.5, 0., 0.7, 2.], device=device))
    dtype = torch.bfloat16 if precision == 'bfloat16' else torch.float32
    model.to(dtype)
    x = torch.randn(2, 83, 32, device=device, dtype=dtype)
    atol = 2e-6 if precision == 'float32' else 0.004
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=precision == 'autocast'):
        _, old = model.step(x[:, :59])
        fast = old
        for t in range(59, 83):
            ref, old = model.step(x[:, t:t+1], old)
            out, fast = model.predict(x[:, t], fast)
            assert out.dtype == ref.dtype and torch.isfinite(out).all()
            torch.testing.assert_close(out, ref[:, 0], atol=atol, rtol=atol)
            assert_state(fast, old, atol=atol, rtol=atol)
