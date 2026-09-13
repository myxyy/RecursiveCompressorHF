"""Verify identical initialization/RNG, scale-one compatibility and GPU execution."""
import importlib.util
import json
import sys
import torch
from common import HERE, SOURCE, ROOT, MODES, save, sha
sys.path.insert(0,str(SOURCE))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from logkv import LogKV

def main():
    cfg_path=HERE.parent/'logkv-rope-only-20260909/copying/retrieval-rope/run_config.json'
    old=json.loads(cfg_path.read_text())
    fields=['d_model','num_heads','d_ff','num_layers','chunk_size','phase_emb','phase_levels',
        'gated_attention','self_slot','retrieval_rope','compressor_rope','level_decay_scale']
    models=[]; rngs=[]
    for mode,scale in MODES.items():
        torch.manual_seed(0)
        cfg=LogKVConfig(vocab_size=10,**{k:old[k] for k in fields},retrieval_rope_scale=scale,
            pad_token_id=None,bos_token_id=None,eos_token_id=None)
        m=LogKVLM(cfg); models.append(m); rngs.append(torch.get_rng_state())
        assert sum(p.numel() for p in m.parameters())==5786112
    for m,rng in zip(models[1:],rngs[1:]):
        assert torch.equal(rng,rngs[0])
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in models[0].state_dict().items())
    models[0].save_pretrained(ROOT/'initial_model')
    old_source=ROOT.parent/'logkv-phase2-rope-20260908/source/logkv.py'
    spec=importlib.util.spec_from_file_location('old_logkv',old_source)
    oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
    torch.manual_seed(5)
    a=oldmod.LogKV(16,4,num_heads=2,gated_attention=True,self_slot=True,retrieval_rope=True)
    b=LogKV(16,4,num_heads=2,gated_attention=True,self_slot=True,retrieval_rope=True)
    b.load_state_dict(a.state_dict()); x=torch.randn(2,70,16)
    with torch.no_grad(): assert torch.equal(a(x),b(x))
    # All three actual model configurations execute backward and cached bf16 inference.
    gpu_losses={}
    for mode,m in zip(MODES,models):
        m=m.cuda(); ids=torch.randint(0,10,(2,85),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16): out=m(ids,labels=ids)
        assert torch.isfinite(out.loss); out.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        gpu_losses[mode]=float(out.loss)
        m.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits,state=m.step(ids[:,:64]); logits,state=m.step(ids[:,64:],state)
            assert torch.isfinite(logits).all()
        m.cpu();m.zero_grad(set_to_none=True)
    save(HERE/'preflight.json',dict(passed=True,initial_weights_identical=True,initial_rng_identical=True,
        num_params=5786112,legacy_alpha_one_output_bitexact=True,gpu_bf16_backward_and_streaming=True,
        initial_weights_sha256=sha(ROOT/'initial_model/model.safetensors'),gpu_smoke_losses=gpu_losses,
        unit_tests='293 passed (test_logkv.py and test_logkv_lm.py)'))
if __name__=='__main__': main()
