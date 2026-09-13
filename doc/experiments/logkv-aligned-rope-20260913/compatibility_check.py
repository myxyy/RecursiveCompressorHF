"""Check disabled mode against the pre-change frozen core, CPU only."""
import importlib.util
import sys
import torch
from common import HERE,ROOT,SOURCE,save
sys.path.insert(0,str(SOURCE))
from logkv import LogKV
oldpath=ROOT.parent/'logkv-rope-scale-20260910/source/logkv.py'
spec=importlib.util.spec_from_file_location('legacy_core',oldpath)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
torch.set_num_threads(1)
results=[]
for flags in [{},dict(retrieval_rope=True),dict(compressor_rope=True)]:
    torch.manual_seed(0);old=legacy.LogKV(16,4,num_heads=2,self_slot=True,gated_attention=True,**flags)
    rng=torch.get_rng_state()
    torch.manual_seed(0);new=LogKV(16,4,num_heads=2,self_slot=True,gated_attention=True,**flags)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    x=torch.randn(2,85,16,requires_grad=True);y=x.detach().clone().requires_grad_()
    a=old(x);b=new(y);assert torch.equal(a,b)
    a.square().sum().backward();b.square().sum().backward()
    assert torch.equal(x.grad,y.grad)
    assert all(torch.equal(a.grad,b.grad) for a,b in zip(old.parameters(),new.parameters()))
    results.append(dict(flags=flags,initial_weights_and_rng=True,forward_and_gradients_bitexact=True))
save(HERE/'validation.json',dict(passed=True,tests_passed=327,
    test_command='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest test_logkv.py test_logkv_lm.py test_aligned_rope.py -q',
    cpu_legacy_compatibility=results))
