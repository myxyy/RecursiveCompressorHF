"""Stateless training path: gradients, precision, and cache routing."""
import copy
from unittest.mock import patch

import pytest
import torch

from models.logkv.attention import CausalConvBlock, LogKV, LogKVBlock
from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM
from models.logkv.pipeline import LogKVLMPipelineStage
from tests.logkv.test_logkv import reference_forward


@pytest.mark.parametrize('chunk', [2, 3, 4])
@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('options', [
    {}, dict(self_slot=True, gated_attention=True),
    dict(self_slot=True, no_level_bias=True),
    dict(phase_emb=True, phase_levels=2, learnable_decay=True, kv_norm=True),
    dict(level_amplify=True, v_norm_only=True),
])
def test_stateless_attention_reference_gradients(chunk, recompute, options):
    torch.manual_seed(81)
    model = LogKV(8, chunk, num_heads=2, **options).double()
    model.recompute_attention = recompute
    x = torch.randn(2, chunk**2 + 1, 8, dtype=torch.float64, requires_grad=True)
    with patch.object(model, 'step', side_effect=AssertionError('forward built a streaming state')):
        actual = model(x)
        grad = torch.autograd.grad(actual.square().sum(), (x, *model.parameters()))
    expected = reference_forward(model, x)
    ref_grad = torch.autograd.grad(expected.square().sum(), (x, *model.parameters()))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(grad, ref_grad):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('length', [1, 16, 17, 2047])
@pytest.mark.parametrize('kernel', [0, 1, 4])
def test_block_step_output_and_gradients(length, kernel):
    torch.manual_seed(82)
    model = LogKVBlock(8, 4, 16, num_heads=2, conv_kernel_size=kernel,
                       self_slot=True, gated_attention=True).double()
    x = torch.randn(1, length, 8, dtype=torch.float64, requires_grad=True)
    actual = model(x)
    grads = torch.autograd.grad(actual.square().sum(), (x, *model.parameters()))
    expected, _ = model.step(x)
    ref_grads = torch.autograd.grad(expected.square().sum(), (x, *model.parameters()))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_lm_and_pipeline_train_without_step_and_preserve_cache_paths():
    config = LogKVConfig(vocab_size=32, d_model=16, num_layers=2, d_ff=32,
                         num_heads=2, conv_kernel_size=4, self_slot=True,
                         gated_attention=True, no_level_bias=True)
    model = LogKVLM(config).double()
    stage = LogKVLMPipelineStage(config, 0, 2, True, True).double()
    stage.load_from_full_model(model.state_dict())
    ids = torch.randint(0, 32, (2, 17))
    ref, hidden = model.step(ids)
    snapshot = copy.deepcopy(model.state_dict())
    with patch.object(LogKV, 'step', side_effect=AssertionError('attention step')), \
         patch.object(CausalConvBlock, 'step', side_effect=AssertionError('conv step')), \
         patch.object(LogKVLM, 'step', side_effect=AssertionError('LM step')):
        actual = model(ids)
        assert actual.past_key_values is None
        torch.testing.assert_close(actual.logits, ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(stage(ids), ref, atol=1e-12, rtol=1e-12)
        actual.logits.square().sum().backward()
    # Existing checkpoint names/values and cached prefill/decode are unchanged.
    torch.testing.assert_close(model.state_dict(), snapshot, atol=0, rtol=0)
    with patch.object(model, 'step', wraps=model.step) as step:
        cached = model(ids, use_cache=True)
        assert step.call_count == 1
        torch.testing.assert_close(cached.past_key_values, hidden, atol=0, rtol=0)
        # Even use_cache=False must consume the supplied past.
        model(ids[:, :2], past_key_values=hidden, use_cache=False)
        assert step.call_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('precision', ['fp32', 'bf16', 'autocast'])
@pytest.mark.parametrize('zero_bias', [False, True])
def test_cuda_block_forward_backward_precision(precision, zero_bias):
    torch.manual_seed(83)
    model = LogKVBlock(32, 4, 64, num_heads=4, conv_kernel_size=4,
                       self_slot=True, gated_attention=True, no_level_bias=zero_bias).cuda()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float32
    model = model.to(dtype=dtype)
    x = torch.randn(2, 67, 32, device='cuda', dtype=dtype, requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'autocast'):
        actual = model(x)
        expected, _ = model.step(x)
    grads = torch.autograd.grad(actual.float().square().mean(), (x, *model.parameters()))
    ref_grads = torch.autograd.grad(expected.float().square().mean(), (x, *model.parameters()))
    tolerance = dict(atol=2e-5, rtol=2e-4) if precision == 'fp32' else dict(atol=2e-3, rtol=0.05)
    torch.testing.assert_close(actual, expected, **tolerance)
    for a, b in zip(grads, ref_grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, **tolerance)
