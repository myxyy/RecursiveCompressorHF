import copy

import pytest
import torch

pytest.importorskip('mamba_ssm')
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from models.mamba2.configuration import Mamba2Config
from models.mamba2.modeling import Mamba2LM


def config(**kw):
    return Mamba2Config(vocab_size=10, d_model=32, num_layers=2,
                        d_state=8, headdim=16, scan_chunk_size=16, **kw)


def test_initialization_is_exactly_upstream():
    torch.manual_seed(7)
    ours = Mamba2LM(config())
    torch.manual_seed(7)
    official = MambaLMHeadModel(config().upstream_config())
    assert ours.state_dict().keys() == official.state_dict().keys()
    for name, value in ours.state_dict().items():
        torch.testing.assert_close(value, official.state_dict()[name], rtol=0, atol=0)
    assert ours.lm_head.weight is ours.backbone.embedding.weight


@pytest.mark.parametrize('ngroups', [1, 2])
def test_reference_chunks_gradients_and_nonmutation(ngroups):
    torch.manual_seed(11)
    model = Mamba2LM(config(ngroups=ngroups)).double()
    other = copy.deepcopy(model)
    ids = torch.randint(0, 10, (2, 37))
    expected, final = model.step(ids)
    outputs, state = [], None
    start = 0
    for length in [1, 2, 3, 17, 14]:
        before = None if state is None else [(a.clone(), b.clone()) for a,b in state]
        out, new_state = other.step(ids[:, start:start+length], state)
        if state is not None:
            for old, snap in zip(state, before):
                for a,b in zip(old, snap):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
        outputs.append(out)
        state, start = new_state, start + length
    actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)
    for a,b in zip(final, state):
        for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10,msg=name)
    for layer in state:
        for tensor in layer:
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    _, later = other.step(ids, state)
    assert sum(t.numel() for layer in later for t in layer) == sum(t.numel() for layer in state for t in layer)


def test_save_load_and_position_aligned_loss(tmp_path):
    model = Mamba2LM(config()).eval()
    ids = torch.randint(0,10,(2,13))
    labels = torch.randint(0,10,(2,13))
    out = model(ids, labels=labels)
    torch.testing.assert_close(out.loss,torch.nn.functional.cross_entropy(out.logits.flatten(0,1),labels.flatten()))
    assert model(ids, labels=torch.full_like(ids,-100)).loss == 0
    model.save_pretrained(tmp_path)
    restored = Mamba2LM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(model(ids).logits,restored(ids).logits,rtol=0,atol=0)
    from inference.predict import _load_model
    loaded = _load_model(str(tmp_path),torch.device('cpu'),torch.float32).eval()
    a,state = loaded.step(ids[:,:4])
    b,state = loaded.predict(ids[:,4],state)
    torch.testing.assert_close(b,loaded(ids).logits[:,4],rtol=1e-5,atol=1e-6)
    generated = loaded.generate(ids[:, :3], max_new_tokens=4, do_sample=False,
                                eos_token_id=None, pad_token_id=0)
    manual = ids[:, :3]
    for _ in range(4):
        manual = torch.cat([manual, loaded(manual).logits[:, -1].argmax(-1)[:, None]], 1)
    torch.testing.assert_close(generated, manual)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for official kernels')
@pytest.mark.parametrize('autocast', [False, True])
def test_gpu_upstream_and_streaming_parity(autocast):
    torch.manual_seed(29)
    model = Mamba2LM(config()).cuda().eval()
    ids = torch.randint(0,10,(2,67),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast):
        official = model(ids).logits
        whole,state = model.step(ids)
        left,st = model.step(ids[:,:19])
        right,st = model.step(ids[:,19:],st)
        ref,ref_state = model.step(ids,reference=True)
    atol,rtol = (0.015,0.05) if autocast else (3e-5,3e-4)
    for actual in (whole,torch.cat([left,right],1),ref):
        torch.testing.assert_close(actual,official,atol=atol,rtol=rtol)
    for actual,expected in zip(st,state):
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    if not autocast:
        # Differentiability of the kernel's initial/final states across a boundary.
        official.square().mean().backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters()}
        model.zero_grad()
        torch.cat([left,right],1).square().mean().backward()
        for n,p in model.named_parameters():
            torch.testing.assert_close(p.grad,grads[n],atol=3e-5,rtol=2e-3,msg=n)


def test_invalid_shapes():
    model = Mamba2LM(config())
    ids = torch.ones(2,5,dtype=torch.long)
    _,state = model.step(ids)
    with pytest.raises(ValueError): model.step(ids[:,:0])
    with pytest.raises(ValueError): model.step(ids,state[:1])
    with pytest.raises(ValueError): model.step(ids[:1],state)
