"""CPU checks for signed learned slopes combined with convolution and streaming."""
import copy
import math
import sys
import torch
from common import HERE, REPO, save
sys.path.insert(0,str(REPO))
from logkv_lm import LogKVLM
from configuration_logkv import LogKVConfig

torch.set_num_threads(1); torch.manual_seed(0)
cfg=dict(vocab_size=10,d_model=16,num_heads=4,d_ff=32,num_layers=2,chunk_size=4,
         conv_kernel_size=4,gated_attention=True,self_slot=True,phase_emb=False)
learn=LogKVLM(LogKVConfig(**cfg,learnable_decay=True)).double().eval()
fixed=LogKVLM(LogKVConfig(**cfg)).double().eval()
fixed.load_state_dict({k:v for k,v in learn.state_dict().items() if not k.endswith('.level_decay')},strict=True)
ids=torch.randint(0,10,(2,67))
with torch.no_grad():
    for layer in learn.layers: layer.attention.level_decay.fill_(math.log(4))
    a,b=learn(ids).logits,fixed(ids).logits
    initial_error=(a-b).abs().max().item()
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    for layer in learn.layers:
        layer.attention.level_decay.copy_(torch.tensor([-.4,0.,.5,1.5],dtype=torch.float64))
    full=learn(ids).logits
    first,h=learn.step(ids[:,:5]);middle,h=learn.step(ids[:,5:65],h)
    one,h=learn.predict(ids[:,65],h);two,h=learn.predict(ids[:,66],h)
    split=torch.cat([first,middle,one[:,None],two[:,None]],1)
    torch.testing.assert_close(full,split,atol=1e-12,rtol=1e-12)
# Finite-difference check includes negative, zero and positive slopes, and the
# activation-checkpoint path used by training.
learn.train(); loss=learn(ids).logits.square().sum();loss.backward()
grads=[l.attention.level_decay.grad.clone() for l in learn.layers]
assert all(torch.isfinite(g).all() and (g.abs()>1e-10).all() for g in grads)
errors=[]
for li,layer in enumerate(learn.layers):
    p=layer.attention.level_decay
    for hi in range(4):
        old=p[hi].item();eps=1e-5
        with torch.no_grad():
            p[hi]=old+eps;plus=learn(ids).logits.square().sum().item()
            p[hi]=old-eps;minus=learn(ids).logits.square().sum().item();p[hi]=old
        numerical=(plus-minus)/(2*eps)
        error=abs(numerical-grads[li][hi].item()); errors.append(error)
        assert error<1e-7
save(HERE/'validation.json',dict(passed=True,initial_fp64_equal_with_exact_log_c=True,
    initial_max_abs_error=initial_error,signed_slope_streaming_max_abs_error=(full-split).abs().max().item(),
    finite_difference_coefficients=8,finite_difference_max_abs_error=max(errors),
    scope='CausalConv plus learned signed head/layer slopes; main model is unmodified.'))
print('Learnable decay + CausalConv: fp64 equivalence, signed streaming and slope gradients passed.')
