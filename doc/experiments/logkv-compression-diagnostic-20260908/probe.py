"""Trace real first-prefix compression trees and real marker readouts, without training."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

SOURCE = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907/source')
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-compression-diagnostic-20260908')
CHECKPOINTS = {
    'phase2': Path('/mnt/raid0/RecursiveCompressor/exp/copying/logkv-d512-logu-ph2-gated-self-refined-20260906/model'),
    'combined-no-decay': Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908/exp/copying/combined-no-decay-fixed10-20260908/model'),
}
sys.path.insert(0,str(SOURCE))
from logkv_lm import LogKVLM


class Trace:
    def __init__(self,model):
        self.arrays = {}
        self.offset = self.length = 0
        self.depths = [0]*len(model.layers)
        self.read_depth = None
        self.condition = 'memory'
        self.enabled = False
        self.batch = 16
        self.pool_checks = []
        for li,layer in enumerate(model.layers):
            att = layer.attention
            att.compressor.register_forward_hook(self.pool_hook(li),with_kwargs=True)
            original = att._attend_levels
            def wrapped(q,n_levels,scale,*flat,li=li,att=att,original=original):
                if self.enabled and self.read_depth is not None:
                    self.readout(att,li,q,n_levels,scale,flat)
                return original(q,n_levels,scale,*flat)
            att._attend_levels = wrapped

    def put(self,key,tensor):
        assert key not in self.arrays,key
        self.arrays[key] = tensor.detach().float().cpu().numpy().copy()

    def pool_hook(self,li):
        def hook(module,args,kwargs,output):
            if not self.enabled or self.read_depth is not None:
                return
            self.depths[li] += 1
            d = self.depths[li]
            if not (self.offset < 4**d <= self.offset+self.length):
                return
            B,H,C,D = self.batch,module.num_heads,4,args[0].shape[-1]
            n = kwargs['num_chunks']
            q,k,v = [x.reshape(B,H,n,C,D)[:,:,0].reshape(B*H,C,D) for x in args]
            qo,ko,vo = [x.reshape(B,H,n,D)[:,:,0] for x in output]
            w = torch.softmax(torch.bmm(q[:,-1:],k.transpose(1,2))*D**-.5,dim=-1)
            # Both checkpoints use content-only compressor weights (no logit decay).
            assert module.decay == 0 and module.raw_decay is None
            prefix = f'{self.condition}_l{li}_d{d}_'
            for name,x in [('q_child',q),('k_child',k),('v_child',v)]:
                self.put(prefix+name,x.reshape(B,H,C,D))
            for name,x in [('q',qo),('k',ko),('v',vo)]:self.put(prefix+name,x)
            self.put(prefix+'weights',w.reshape(B,H,C))
            for name,x,out in [('k',k,ko),('v',v,vo)]:
                pre = torch.bmm(w,x).reshape(B,H,D)
                transformed = module.transform_positions(x,1) if module.position_vectors is not None else x
                post = torch.bmm(w,transformed).reshape(B,H,D)
                self.put(prefix+name+'_pre_transform',pre)
                self.pool_checks.append(float((post.float()-out.float()).abs().max()))
                # Reconstructed subset may select another GEMM kernel, allowing bf16 rounding.
                torch.testing.assert_close(post.float(),out.float(),rtol=.025,atol=.025)
        return hook

    def readout(self,att,li,q,n,scale,flat):
        B,H,D = self.batch,att.num_heads,att.head_dim
        ks,vs,locals_,invalids = flat[:n],flat[n:2*n],flat[2*n:3*n],flat[3*n:4*n]
        logits,raw = [],[]
        for i in range(n):
            slot = ks[i][:,locals_[i],:]
            raw.append(torch.einsum('bld,blcd->blc',q,slot)*scale)
            if att.relative_key is not None:
                j = (~invalids[i]).sum(-1)
                dist = (j[:,None]-torch.arange(3,device=q.device)[None]).clamp(0,3)
                slot = slot + att.relative_key[:,dist].repeat(B,1,1,1).to(slot.dtype)
            eff = torch.einsum('bld,blcd->blc',q,slot)*scale
            eff = eff - att.level_decay_scale*i*math.log(4)
            logits.append(eff.masked_fill(invalids[i][None],float('-inf')))
        sk = flat[4*n]
        if att.relative_key is not None:
            sk = sk + att.relative_key[:,0].repeat(B,1)[:,None].to(sk.dtype)
        logits.append(((q*sk).sum(-1)*scale).unsqueeze(-1))
        all_logits = torch.cat(logits,-1).float()
        index = self.read_depth*3
        assert torch.isfinite(all_logits[:,:,index]).all()
        mem_logit = all_logits[:,:,index]
        others = all_logits.clone();others[:,:,index] = -float('inf')
        pref = f'{self.condition}_l{li}_d{self.read_depth}_read_'
        def shaped(x):return x.reshape(B,H,*x.shape[1:])
        self.put(pref+'q',shaped(q))
        self.put(pref+'memory_raw_logit',shaped(raw[self.read_depth][:,:,0]))
        self.put(pref+'memory_effective_logit',shaped(mem_logit))
        self.put(pref+'memory_vs_max_other',shaped(mem_logit-others.max(-1).values))
        self.put(pref+'memory_weight',shaped(all_logits.softmax(-1)[:,:,index]))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=CHECKPOINTS,required=True)
    p.add_argument('--precision',choices=['bf16','fp32'],required=True)
    args=p.parse_args()
    out=ROOT/f'{args.mode}-{args.precision}';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    # fp32 uses IEEE matmul (TF32 disabled); bf16 matches the prior explicit dtype/autocast protocol.
    torch.backends.cuda.matmul.allow_tf32=False
    model=LogKVLM.from_pretrained(CHECKPOINTS[args.mode]).to(
        device='cuda',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float32).eval()
    trace=Trace(model)
    digits=torch.randint(1,9,(16,10),generator=torch.Generator().manual_seed(12345)).cuda()
    metadata=dict(mode=args.mode,precision=args.precision,samples=16,seed=12345,depths=list(range(2,9)),
                  prefix='digits at positions 0..9, then blanks to N=4**depth; 11 marker tokens',
                  blank_control='all-zero prefix at the identical positions, then the same markers',
                  checkpoint=str(CHECKPOINTS[args.mode]),
                  checkpoint_sha256=hashlib.sha256((CHECKPOINTS[args.mode]/'model.safetensors').read_bytes()).hexdigest(),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,
                  tf32=False,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=False,results=[])
    def call(x,hidden=None):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.precision=='bf16'):
            return model.step(x,hidden)
    # Observational hooks and readout reconstruction must leave outputs unchanged.
    x=torch.zeros(16,16,dtype=torch.long,device='cuda');x[:,:10]=digits
    marker=torch.full((16,11),9,dtype=torch.long,device='cuda')
    _,h=call(x);ref,_=call(marker,h)
    trace.enabled=True;trace.offset=0;trace.length=16;trace.depths=[0,0]
    _,h=call(x);trace.read_depth=2;observed,_=call(marker,h)
    assert torch.equal(ref,observed),'Instrumentation changed model outputs'
    metadata['instrumentation_output_exact']=True
    trace.arrays.clear();trace.pool_checks.clear();trace.read_depth=None
    start=time.monotonic()
    trace.put('digits',digits)
    for condition in ('memory','blank'):
        trace.condition=condition;trace.read_depth=None
        h=None;offset=0
        for d in range(2,9):
            N=4**d
            while offset<N:
                length=min(4096,N-offset)
                x=torch.zeros(16,length,dtype=torch.long,device='cuda')
                if offset==0 and condition=='memory':x[:,:10]=digits
                trace.offset=offset;trace.length=length;trace.depths=[0,0]
                _,h=call(x,h);offset+=length
            for li,state in enumerate(h):
                assert state[1]==N and state[0][d][1].shape[1]==1
                for index,name in enumerate(('q','k','v')):
                    value=state[0][d][index].reshape(16,8,64).float().cpu().numpy()
                    np.testing.assert_array_equal(value,trace.arrays[f'{condition}_l{li}_d{d}_{name}'])
            trace.read_depth=d
            logits,_=call(marker,h)
            trace.read_depth=None
            trace.put(f'{condition}_d{d}_answer_logits',logits[:,-10:])
            pred=logits[:,-10:].argmax(-1)
            if condition=='memory':
                good=pred==digits
                correct=logits[:,-10:].float().gather(-1,digits[:,:,None]).squeeze(-1)
                other=logits[:,-10:].float().clone().scatter(-1,digits[:,:,None],-float('inf'))
                margin=correct-other.max(-1).values
                trace.put(f'memory_d{d}_answer_margin',margin)
                r=dict(depth=d,N=N,T=N-9,token_correct=int(good.sum()),string_correct=int(good.all(-1).sum()),n=16,
                       correct_logit_margin_mean=float(margin.mean()),correct_logit_margin_min=float(margin.min()))
                metadata['results'].append(r);print(args.mode,args.precision,r,flush=True)
            assert all(state[1]==N for state in h),'Answer branch mutated the prefix state'
    for li,layer in enumerate(model.layers):
        comp=layer.attention.compressor
        if comp.position_vectors is not None:
            for name in ('position_vectors','position_permutations'):
                trace.put(f'parameters_l{li}_{name}',getattr(comp,name))
        if layer.attention.relative_key is not None:
            trace.put(f'parameters_l{li}_relative_key',layer.attention.relative_key)
    metadata.update(complete=True,elapsed_sec=time.monotonic()-start,
                    finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    pooling_reconstruction_max_abs=max(trace.pool_checks),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    np.savez_compressed(out/'trace.npz',**trace.arrays)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('COMPLETE',out,metadata['elapsed_sec'],flush=True)


if __name__=='__main__':main()
