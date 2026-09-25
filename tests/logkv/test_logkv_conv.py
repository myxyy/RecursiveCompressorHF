"""Causal convolution ordering, streaming storage/gradients and LM integration."""
import copy
import json

import pytest
import torch
import torch.nn.functional as F

from models.logkv.configuration import LogKVConfig
from models.logkv.attention import CausalConvBlock, LogKVBlock
from models.logkv.modeling import LogKVLM
from tests.logkv.test_logkv import reference_forward


def conv_reference(module, x):
    """Explicit past-only taps, independent of Conv1d and its padding path."""
    h = module.norm(x)
    k = module.kernel_size
    padded = torch.cat([h.new_zeros(h.size(0), k-1, h.size(2)), h], dim=1)
    y = module.conv.bias[None, None, :].expand_as(h)
    for tap in range(k):
        y = y + padded[:, tap:tap+x.size(1)] * module.conv.weight[:, 0, tap]
    return x + F.silu(y)


@pytest.mark.parametrize('kernel', [1, 4, 7])
def test_convolution_reference_streaming_and_bounded_storage(kernel):
    torch.manual_seed(1)
    conv = CausalConvBlock(8, kernel).double()
    x = torch.randn(2, 35, 8, dtype=torch.double)
    with torch.no_grad():
        ref = conv_reference(conv, x)
        parts, cache, start = [], None, 0
        for size in [1, 2, 13, 1, 18]:
            before = None if cache is None else cache.clone()
            old = cache
            y, cache = conv.step(x[:, start:start+size], cache)
            if old is not None:
                assert torch.equal(old, before)
            assert cache.shape == (2, kernel-1, 8)
            assert cache.untyped_storage().nbytes() == cache.numel()*cache.element_size()
            parts.append(y); start += size
        torch.testing.assert_close(torch.cat(parts, 1), ref, atol=1e-12, rtol=1e-12)
        empty, unchanged = conv.step(x[:, :0], cache)
        assert empty.shape[1] == 0 and torch.equal(cache, unchanged)


@pytest.mark.parametrize('self_slot', [False, True])
@pytest.mark.parametrize('phase', [False, True])
def test_block_independent_reference_causality_and_predict(self_slot, phase):
    torch.manual_seed(2)
    block = LogKVBlock(8, 4, 16, num_heads=2, self_slot=self_slot,
                      gated_attention=True, phase_emb=phase, phase_levels=2,
                      conv_kernel_size=4).double().eval()
    x = torch.randn(2, 67, 8, dtype=torch.double)
    with torch.no_grad():
        v = conv_reference(block.causal_conv, x)
        v = v + reference_forward(block.attention, block.attention_norm(v))
        ref = v + block.ffn(block.ffn_norm(v))
        torch.testing.assert_close(block(x), ref, atol=1e-12, rtol=1e-12)
        first, hidden = block.step(x[:, :3])
        saved_cache = hidden[0].clone()
        rest, last = block.step(x[:, 3:65], hidden)
        assert torch.equal(hidden[0], saved_cache)
        token1, last = block.predict(x[:, 65], last)
        token2, last = block.predict(x[:, 66], last)
        torch.testing.assert_close(torch.cat([first, rest, token1[:, None], token2[:, None]], 1),
                                   ref, atol=1e-12, rtol=1e-12)
        changed = x.clone(); changed[:, 33:] += torch.randn_like(changed[:, 33:])
        torch.testing.assert_close(block(changed)[:, :33], ref[:, :33], atol=1e-12, rtol=1e-12)


def test_split_backward_matches_full_backward():
    torch.manual_seed(3)
    full = LogKVBlock(8, 4, 16, num_heads=2, gated_attention=True,
                      self_slot=True, conv_kernel_size=4).double()
    split = copy.deepcopy(full)
    x = torch.randn(2, 23, 8, dtype=torch.double, requires_grad=True)
    xs = x.detach().clone().requires_grad_(True)
    full(x).square().sum().backward()
    a, h = split.step(xs[:, :2]); b, h = split.step(xs[:, 2:17], h)
    c, h = split.step(xs[:, 17:], h)
    torch.cat([a, b, c], 1).square().sum().backward()
    torch.testing.assert_close(xs.grad, x.grad, atol=1e-10, rtol=1e-10)
    for (name, p), (_, ps) in zip(full.named_parameters(), split.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(ps.grad, p.grad, atol=1e-10, rtol=1e-10)


def test_conv_lm_save_load_generate_and_precision(tmp_path):
    torch.manual_seed(4)
    cfg = LogKVConfig(vocab_size=10, d_model=16, num_heads=4, d_ff=32,
        num_layers=2, conv_kernel_size=4, gated_attention=True, self_slot=True,
        bos_token_id=None, eos_token_id=None, pad_token_id=None)
    model = LogKVLM(cfg).double().eval()
    ids = torch.randint(0, 10, (2, 37))
    with torch.no_grad():
        ref = model(ids).logits
        h, parts = None, []
        for token in ids.unbind(1):
            y, h = model.predict(token, h); parts.append(y[:, None])
        torch.testing.assert_close(torch.cat(parts, 1), ref, atol=1e-12, rtol=1e-12)
        for history, _ in h:
            assert history.shape == (2, 3, 16)
    model.save_pretrained(tmp_path / 'model')
    loaded = LogKVLM.from_pretrained(tmp_path / 'model').double().eval()
    assert loaded.config.conv_kernel_size == 4
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, ref, atol=1e-12, rtol=1e-12)
        generated = loaded.generate(ids[:1, :7], max_new_tokens=5, do_sample=False)
        logits, h = loaded.step(ids[:1, :7]); tokens = []
        for _ in range(5):
            token = logits[:, -1].argmax(-1); tokens.append(token[:, None])
            logits, h = loaded.step(token[:, None], h)
        assert torch.equal(generated[:, 7:], torch.cat(tokens, 1))
        bf = loaded.bfloat16()(ids).logits
        assert bf.dtype == torch.bfloat16 and torch.isfinite(bf).all()


def test_old_configuration_and_weights_load_without_convolution(tmp_path):
    cfg = LogKVConfig(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=2)
    model = LogKVLM(cfg).eval()
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    del config['conv_kernel_size']
    (tmp_path / 'config.json').write_text(json.dumps(config))
    loaded = LogKVLM.from_pretrained(tmp_path).eval()
    assert loaded.config.conv_kernel_size == 0
    assert all(layer.causal_conv is None for layer in loaded.layers)
    assert loaded.state_dict().keys() == model.state_dict().keys()
    ids = torch.tensor([[1, 4, 2, 8, 0]])
    with torch.no_grad():
        assert torch.equal(loaded(ids).logits, model(ids).logits)


def test_conv_weights_use_adamw_not_muon():
    from training.train_logkv import split_params_for_muon
    model = LogKVLM(LogKVConfig(vocab_size=10, d_model=8, num_heads=2,
        d_ff=16, num_layers=2, conv_kernel_size=4))
    muon, adamw = split_params_for_muon(model)
    assert muon and all(p.ndim == 2 for p in muon)
    adamw_ids = {id(p) for p in adamw}
    for layer in model.layers:
        assert id(layer.causal_conv.conv.weight) in adamw_ids
    assert len({id(p) for p in muon + adamw}) == len(list(model.parameters()))


@pytest.mark.parametrize('kernel', [-1, 1.5])
def test_invalid_kernel_rejected(kernel):
    with pytest.raises(ValueError):
        LogKVConfig(conv_kernel_size=kernel)
