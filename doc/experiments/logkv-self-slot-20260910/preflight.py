"""CPU checks: identical initialization/RNG and independent no-self oracle."""
import json
import subprocess
import sys
import torch
from common import HERE, SOURCE, SOURCE_COMMIT, BASELINE, REPO, save, sha
sys.path.insert(0, str(SOURCE))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from logkv import LogKV
sys.path.append(str(REPO))
from test_logkv import reference_forward

def main():
    torch.set_num_threads(1)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip() == SOURCE_COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    assert sha(SOURCE/'logkv.py') == sha(REPO/'logkv.py')
    cfg = json.loads((BASELINE/'model/config.json').read_text())
    assert cfg['self_slot'] and not cfg['phase_emb'] and cfg['retrieval_rope']
    states=[]; rngs=[]
    for enabled in (True, False):
        torch.manual_seed(0)
        m=LogKVLM(LogKVConfig.from_dict({**cfg,'self_slot':enabled}))
        states.append({k:v.clone() for k,v in m.state_dict().items()})
        rngs.append(torch.get_rng_state().clone())
        assert all(layer.attention.self_slot == enabled and layer.attention.phase_levels == 0 for layer in m.layers)
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(*rngs)
    # Independent materialized attention oracle checks the newly used flag pair,
    # including no-KV position zero, C^2 boundary, gradients and streaming.
    torch.manual_seed(29)
    m=LogKV(16,4,num_heads=2,phase_emb=False,gated_attention=True,
            self_slot=False,retrieval_rope=True).double()
    x=torch.randn(2,69,16,dtype=torch.float64,requires_grad=True)
    expected=reference_forward(m,x)
    expected.square().sum().backward()
    grads=[x.grad.clone()]+[p.grad.clone() for p in m.parameters()]
    m.zero_grad(); x.grad=None
    actual=m(x)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    actual.square().sum().backward()
    for a,b in zip([x.grad]+[p.grad for p in m.parameters()],grads):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    with torch.no_grad():
        parts=[];hidden=None
        for a,b in zip((0,1,3,4,15,16,63,64),(1,3,4,15,16,63,64,69)):
            y,hidden=m.step(x[:,a:b],hidden);parts.append(y)
        torch.testing.assert_close(torch.cat(parts,1),expected,atol=1e-12,rtol=1e-12)
    save(HERE/'preflight.json',dict(source_commit=SOURCE_COMMIT,
         parameter_count=sum(v.numel() for v in states[0].values()),
         initial_weights_identical=True,initial_rng_state_identical=True,
         phase_disabled_both=True,no_self_reference_gradients_streaming_passed=True,
         training_data_seed=1,training_seed=0,original_initial_checkpoint_available=False,
         initialization_note='Reconstructed from frozen code and saved config; original step-zero weights were not archived'))
    print('Preflight passed: identical reconstructed weights/RNG; no-self oracle/gradients/streaming')

if __name__=='__main__': main()
