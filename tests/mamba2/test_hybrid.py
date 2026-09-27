import copy
import pytest
import torch

pytest.importorskip('mamba_ssm')
from models.mamba2.modeling import Mamba2LM
from models.mamba2.configuration import Mamba2Config
from models.mamba2_logkv.configuration import Mamba2LogKVConfig
from models.mamba2_logkv.modeling import Mamba2LogKVLM


def config():
    return Mamba2LogKVConfig(vocab_size=10, d_model=32, num_layers=2, num_logkv_layers=2,
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


def test_official_prefix_initialization_and_no_duplicate_conv():
    torch.manual_seed(11)
    hybrid = Mamba2LogKVLM(config())
    torch.manual_seed(11)
    baseline = Mamba2LM(Mamba2Config(vocab_size=10, d_model=32, num_layers=2,
                                   d_state=8, headdim=16, scan_chunk_size=16))
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(hybrid.state_dict()[name], value, atol=0, rtol=0)
    assert hybrid.lm_head.weight is hybrid.backbone.embedding.weight
    assert all(block.mixer.d_conv == 4 for block in hybrid.backbone.layers)
    assert all(block.causal_conv is None for block in hybrid.logkv_layers)


def test_streaming_gradient_nonmutation_and_logarithmic_state():
    torch.manual_seed(19)
    model = Mamba2LogKVLM(config()).double().eval()
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
    for levels, offset in st['logkv']:
        assert offset == ids.shape[1]
        assert sum(level[0].shape[1] for level in levels) == 5  # 83 = 1*64+1*16+0*4+3
        assert all(level[0].shape[1] < 4 for level in levels)
    assert all(t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in tensors(st))


def test_save_load_generate_and_invalid_state(tmp_path):
    model = Mamba2LogKVLM(config()).eval()
    ids = torch.randint(0, 10, (2, 9))
    labels = torch.randint(0, 10, ids.shape)
    actual = model(ids, labels=labels)
    torch.testing.assert_close(actual.loss, torch.nn.functional.cross_entropy(
        actual.logits.flatten(0, 1), labels.flatten()))
    model.save_pretrained(tmp_path)
    restored = Mamba2LogKVLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(ids).logits, actual.logits, rtol=0, atol=0)
    assert restored.lm_head.weight is restored.backbone.embedding.weight
    from inference.predict import _load_model
    loaded = _load_model(str(tmp_path), torch.device('cpu'), torch.float32).eval()
    generated = loaded.generate(ids[:, :3], max_new_tokens=4, do_sample=False,
                                eos_token_id=None, pad_token_id=0)
    manual = ids[:, :3]
    for _ in range(4):
        manual = torch.cat([manual, loaded(manual).logits[:, -1].argmax(-1)[:, None]], 1)
    torch.testing.assert_close(generated, manual)
    with pytest.raises(ValueError): loaded.step(ids, {'mamba': [], 'logkv': []})
    with pytest.raises(ValueError): Mamba2LogKVConfig(conv_kernel_size=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('precision', ['fp32', 'autocast', 'bf16'])
def test_gpu_training_streaming_and_gradients(precision):
    torch.manual_seed(37)
    model = Mamba2LogKVLM(config()).cuda().eval()
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
