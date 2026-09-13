"""CPU independent-oracle and live GPU smoke checks, no training."""
import argparse,json,sys
import torch
from common import HERE,ROOT,SOURCE,save
from probe import Probe,tail,clone_saved
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM


def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(47)
 device=torch.device(args.device)
 dtype=torch.float64 if device.type=='cpu' else torch.bfloat16
 precision='fp32' if device.type=='cpu' else 'bf16'
 # Eight heads keeps the candidate head index identical to the full model.
 m=LogKVLM(LogKVConfig(vocab_size=10,d_model=32,num_heads=8,d_ff=64,num_layers=2,chunk_size=4,
          phase_emb=False,retrieval_rope=True,self_slot=True,gated_attention=True)).to(device=device,dtype=dtype).eval()
 probe=Probe(m);memory=torch.randint(1,9,(2,10));states={0:None}
 a,_=tail(m,probe,memory,16,states,precision,observe=False)
 b,donor=tail(m,probe,memory,16,states,precision)
 torch.testing.assert_close(a,b,atol=0,rtol=0)
 for patch in ('sham_head8','sham_gate','sham_ffn','sham_block'):
  out,_=tail(m,probe,memory,16,states,precision,patch=patch,donor=donor)
  torch.testing.assert_close(out,a,atol=0,rtol=0)
 altered=clone_saved(donor);altered['l1_raw'][:,7]+=2
 out,saved=tail(m,probe,memory,16,states,precision,patch='head8',donor=altered)
 torch.testing.assert_close(out[:,:8],a[:,:8],atol=0,rtol=0)
 torch.testing.assert_close(saved['l1_raw'][:,7],altered['l1_raw'][:,7],atol=0,rtol=0)
 assert not torch.equal(out[:,8:],a[:,8:])
 # Materialized independent softmax of synthetic 8-level logits verifies
 # score interventions and the untouched heads, including all-masked levels.
 att=m.layers[0].attention;H=8;D=4;B=2;n=8
 q=torch.randn(B*H,1,D,device=device,dtype=dtype)
 ks=[torch.randn(B*H,3,D,device=device,dtype=dtype) for _ in range(n)]
 vs=[torch.randn_like(k) for k in ks]
 loc=[torch.tensor([[0,1,2]],device=device) for _ in range(n)]
 phases=[2,0,1,0,0,0,0,3]
 masks=[torch.arange(3,device=device)[None]>=j for j in phases]
 selfs=(torch.randn_like(q),torch.randn_like(q));flat=(*ks,*vs,*loc,*masks,*selfs)
 from logkv import _local_rope
 from math import log
 with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
  probe.active=False;base=att._attend_levels(q,n,D**-.5,*flat)
  for patch in ('sham_phase','phase','mask','phase_mask'):
   probe.set(0,patch);actual=att._attend_levels(q,n,D**-.5,*flat)
   logits=[];values=[]
   for i in range(n):
    k=ks[i];rot=_local_rope(k,torch.arange(3,device=device)[None]-phases[i])
    if i==7 and patch in ('phase','phase_mask'):
     rot=rot.clone();rot[7::8,0]=_local_rope(k[7::8,0],q.new_tensor(-2))
    logit=torch.einsum('bld,bcd->blc',q,rot)*D**-.5-i*log(4)
    logit=logit.masked_fill(masks[i][None],float('-inf'))
    if i==7 and patch in ('mask','phase_mask'):logit[7::8,0,2]=float('-inf')
    logits.append(logit);values.append(vs[i])
   logits.append((q*selfs[0]).sum(-1)[...,None]*D**-.5);values.append(selfs[1])
   ls=torch.cat(logits,-1)
   prob=ls.double().softmax(-1) if dtype==torch.float64 else ls.float().softmax(-1)
   expected=torch.einsum('blc,bcd->bld',prob.to(dtype),torch.cat(values,1))
   if device.type=='cpu':torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
   else:torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)
   if patch=='sham_phase':torch.testing.assert_close(actual,base,atol=0,rtol=0)
 probe.close()
 save(HERE/f'preflight-{device.type}.json',dict(device=str(device),observation_identity=True,
      sham_patches_identity=True,single_position_patch_locality=True,score_interventions_independent_oracle=True))
 print('Preflight passed',device)

if __name__=='__main__':main()
