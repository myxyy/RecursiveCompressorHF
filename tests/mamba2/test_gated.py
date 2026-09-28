import copy
import pytest
import torch

pytest.importorskip('mamba_ssm')
from models.mamba2.modeling import Mamba2LM
from models.mamba2.configuration import Mamba2Config
from models.mamba2_logkv_gated.configuration import GatedMambaLogKVConfig
from models.mamba2_logkv_gated.modeling import GatedMambaLogKVLM
from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM


def config():
    return GatedMambaLogKVConfig(vocab_size=10, d_model=32, num_layers=2, num_mamba_layers=2,
                            d_state=8, headdim=16, scan_chunk_size=16, num_heads=2,
                            d_ff=64, chunk_size=4, pad_token_id=None, eos_token_id=None)


def tensors(state):
    if isinstance(state, torch.Tensor):
        yield state
    elif isinstance(state, dict):
        for v in state.values():
            yield from tensors(v)
    elif isinstance(state, (tuple, list)):
        for v in state:
            yield from tensors(v)


def test_zero_gate_exactly_recovers_standalone_logkv():
    torch.manual_seed(11)
    model = GatedMambaLogKVLM(config()).eval()
    torch.manual_seed(11)
    base = LogKVLM(LogKVConfig(vocab_size=10, d_model=32, num_layers=2, num_heads=2,
                              d_ff=64, chunk_size=4, conv_kernel_size=4,
                              gated_attention=True, self_slot=True, pad_token_id=None, eos_token_id=None)).eval()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    ids = torch.randint(0, 10, (2, 29))
    torch.testing.assert_close(model(ids).logits, base(ids).logits, rtol=0, atol=0)
    assert all(layer.causal_conv is not None for layer in model.layers)
    assert model.head.weight is not model.embedding.weight
    assert model.mamba.lm_head.weight is model.mamba.backbone.embedding.weight
    loss = model(ids, labels=ids).loss
    loss.backward()
    assert model.mamba_gate.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() == 0 for p in model.mamba.parameters())
    model.zero_grad()
    with torch.no_grad(): model.mamba_gate.fill_(0.1)
    model(ids, labels=ids).loss.backward()
    assert any(p.grad.abs().sum() > 0 for p in model.mamba.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_zero_gate_gpu_autocast_matches_logkv():
    model = GatedMambaLogKVLM(config()).cuda().eval()
    base = LogKVLM(LogKVConfig(vocab_size=10, d_model=32, num_layers=2, num_heads=2,
                              d_ff=64, chunk_size=4, conv_kernel_size=4,
                              gated_attention=True, self_slot=True)).cuda().eval()
    base.load_state_dict({k: model.state_dict()[k] for k in base.state_dict()})
    ids = torch.randint(0, 10, (2, 67), device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(model(ids).logits, base(ids).logits, rtol=0, atol=0)


def test_streaming_gradient_nonmutation_and_logarithmic_state():
    torch.manual_seed(19)
    model = GatedMambaLogKVLM(config()).double().eval()
    with torch.no_grad(): model.mamba_gate.fill_(0.1)
    other = copy.deepcopy(model)
    ids = torch.randint(0, 10, (2, 83))
    full, state = model.step(ids)
    chunks, st, start = [], None, 0
    for length in [1, 2, 13, 17, 50]:
        snap = [t.clone() for t in tensors(st)]
        y, nxt = other.step(ids[:, start:start+length], st)
        for t, expected in zip(tensors(st), snap):
            torch.testing.assert_close(t, expected, rtol=0, atol=0)
        chunks.append(y)
        st, start = nxt, start + length
    out = torch.cat(chunks, 1)
    torch.testing.assert_close(out, full, rtol=1e-9, atol=1e-10)
    for x, y in zip(tensors(state), tensors(st)):
        torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-10)
    full.square().mean().backward()
    out.square().mean().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-7, atol=1e-9, msg=name)
    for conv, (levels, offset) in st['logkv']:
        assert conv.shape[1] == 3
        assert offset == ids.shape[1]
        assert sum(level[0].shape[1] for level in levels) == 5  # 83 = 1*64+1*16+0*4+3
        assert all(level[0].shape[1] < 4 for level in levels)
    assert all(t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in tensors(st))


def test_save_load_generate_and_invalid_state(tmp_path):
    model = GatedMambaLogKVLM(config()).eval()
    with torch.no_grad(): model.mamba_gate.fill_(0.1)
    ids = torch.randint(0, 10, (2, 9))
    labels = torch.randint(0, 10, ids.shape)
    actual = model(ids, labels=labels)
    torch.testing.assert_close(actual.loss, torch.nn.functional.cross_entropy(
        actual.logits.flatten(0, 1), labels.flatten()))
    model.save_pretrained(tmp_path)
    restored = GatedMambaLogKVLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(ids).logits, actual.logits, rtol=0, atol=0)
    assert restored.mamba.lm_head.weight is restored.mamba.backbone.embedding.weight
    from inference.predict import _load_model
    loaded = _load_model(str(tmp_path), torch.device('cpu'), torch.float32).eval()
    generated = loaded.generate(ids[:, :3], max_new_tokens=4, do_sample=False,
                                eos_token_id=None, pad_token_id=0)
    manual = ids[:, :3]
    for _ in range(4):
        manual = torch.cat([manual, loaded(manual).logits[:, -1].argmax(-1)[:, None]], 1)
    torch.testing.assert_close(generated, manual)
    with pytest.raises(ValueError): loaded.step(ids, {'mamba': [], 'logkv': []})
    with pytest.raises(ValueError): GatedMambaLogKVConfig(conv_kernel_size=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('precision', ['fp32', 'autocast', 'bf16'])
def test_gpu_training_streaming_and_gradients(precision):
    torch.manual_seed(37)
    model = GatedMambaLogKVLM(config()).cuda().eval()
    with torch.no_grad(): model.mamba_gate.fill_(0.1)
    if precision == 'bf16':
        model = model.bfloat16()
    ids = torch.randint(0, 10, (2, 67), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision != 'fp32'):
        full = model(ids).logits
        whole, state = model.step(ids)
        a, st = model.step(ids[:, :19])
        b, st = model.step(ids[:, 19:], st)
    atol, rtol = (3e-4, 2e-3) if precision == 'fp32' else (0.03, 0.08)
    for out in [whole, torch.cat([a, b], 1)]:
        torch.testing.assert_close(out, full, atol=atol, rtol=rtol)
    if precision == 'fp32':
        full.square().mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad()
        torch.cat([a, b], 1).square().mean().backward()
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.grad, grads[n], atol=3e-4, rtol=3e-3, msg=n)
