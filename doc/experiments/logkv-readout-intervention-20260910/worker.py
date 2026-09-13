"""One checkpoint/precision lane; reusable prefixes, matched donor memories."""
import argparse,json,sys,time
import numpy as np
import torch
from common import HERE,ROOT,BASELINE,OLD,JOBS,save,sha,now
from probe import Probe,prefix_states,tail
from logkv_lm import LogKVLM

PATCHES=('sham_head8','head8','head1','l1_block','sham_phase','phase','mask','phase_mask',
         'sham_gate','l2_gate','l2_raw','l2_raw_gate','sham_ffn','l2_ffn','sham_block')


def tree_clone(x):
 if torch.is_tensor(x):return x.clone()
 if isinstance(x,dict):return {k:tree_clone(v) for k,v in x.items()}
 if isinstance(x,list):return [tree_clone(t) for t in x]
 if isinstance(x,tuple):return tuple(tree_clone(t) for t in x)
 return x

def tree_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
 return a==b


def main():
 p=argparse.ArgumentParser();p.add_argument('--job',choices=[j['name'] for j in JOBS],required=True);args=p.parse_args()
 job=next(j for j in JOBS if j['name']==args.job);out=ROOT/job['name'];out.mkdir(exist_ok=False)
 torch.set_num_threads(1)
 precision=job['precision'];cp=job['checkpoint']
 torch.set_float32_matmul_precision('highest' if precision=='fp32' else 'high')
 if precision=='fp32':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 path=BASELINE/('model_best' if cp=='best' else 'model');digest=sha(path/'model.safetensors')
 model=LogKVLM.from_pretrained(path).to(device='cuda:0',dtype=torch.float32 if precision=='fp32' else torch.bfloat16).eval()
 assert model.config.self_slot and not model.config.phase_emb and model.config.retrieval_rope
 assert not model.config.compressor_rope and model.config.level_decay_scale==1
 probe=Probe(model);collected={};case_meta={};checks=[];started=time.monotonic()
 report=dict(state='running',started=now(),job=job,weight_sha256=digest,case_groups=0)
 save(out/'status.json',report)
 try:
  for cohort,seed,count in [('discovery',4321,16),('validation',20260911,32)]:
   memory=torch.randint(1,9,(count,10),generator=torch.Generator().manual_seed(seed))
   for begin in range(0,count,16):
    mem=memory[begin:begin+16];probe.active=False;t0=time.monotonic()
    states=prefix_states(model,mem,precision);original=tree_clone(states)
    for layer in (0,1):
     for kv in (1,2):
      assert torch.equal(states[32768][layer][0][7][kv][:,0],states[49152][layer][0][7][kv][:,0])
    print(cohort,begin,'prefix seconds',round(time.monotonic()-t0,2),flush=True)
    bases={}
    def record(name,T,digit,patch,logits,trace,donorT=None):
     key=cohort+'__'+name
     meta=dict(cohort=cohort,seed=seed,T=T,digit=digit+1,patch=patch,donor_T=donorT,
               absolute_position=T+10+digit,checkpoint=cp,precision=precision)
     if key in case_meta:assert case_meta[key]==meta
     case_meta[key]=meta
     arrays={'logits':logits.cpu().numpy(),'target':mem.numpy()}
     for k,v in trace.items():arrays[k]=v.float().cpu().numpy()
     collected.setdefault(key,[]).append(arrays)
    def baseline(T,digit=8):
     key=(T,digit)
     if key not in bases:
      logits,trace=tail(model,probe,mem,T,states,precision,digit)
      plain,_=tail(model,probe,mem,T,states,precision,digit,observe=False)
      torch.testing.assert_close(logits,plain,atol=0,rtol=0)
      bases[key]=(logits,trace)
      record(f'T{T}-d{digit+1}-baseline',T,digit,'none',logits,trace)
     return bases[key]
    horizons=range(49144,49161) if precision=='bf16' else (49151,49152,49153)
    for T in horizons:baseline(T,49170-T-10 if T in (49151,49152,49153) else 8)
    # Compare the same answer index and same lower base-4 phases. The good
    # horizon differs by exactly 4^7; retained memory K/V equality is checked.
    for badT,digit in [(49152,8),(49153,7)]:
     goodT=badT-16384;good,good_trace=baseline(goodT,digit);bad,bad_trace=baseline(badT,digit)
     for patch in PATCHES:
      donor=bad_trace if patch.startswith('sham') else good_trace
      logits,trace=tail(model,probe,mem,badT,states,precision,digit,patch,donor)
      torch.testing.assert_close(logits[:,:digit],bad[:,:digit],atol=0,rtol=0)
      if patch.startswith('sham'):torch.testing.assert_close(logits,bad,atol=0,rtol=0)
      record(f'T{badT}-d{digit+1}-{patch}',badT,digit,patch,logits,trace,goodT if not patch.startswith('sham') else badT)
     for patch in ('head8','head1','l1_block','l2_gate','l2_raw','l2_raw_gate','l2_ffn'):
      logits,trace=tail(model,probe,mem,goodT,states,precision,digit,patch,bad_trace)
      torch.testing.assert_close(logits[:,:digit],good[:,:digit],atol=0,rtol=0)
      record(f'T{goodT}-d{digit+1}-reverse-{patch}',goodT,digit,'reverse-'+patch,logits,trace,badT)
    assert tree_equal(states,original)
    if cohort=='discovery':
     for T in (49151,49152,49153):
      if precision=='fp32' and T!=49152:continue
      digit=49170-T-10
      logits,_=baseline(T,digit)
      ref=np.load(OLD/f'self-on-{cp}-{precision}-paired-T{T}.npz')
      assert np.array_equal(mem.numpy(),ref['target'])
      # Exact predictions are mandatory; raw logits may depend on TF32 kernels
      # only in fp32, so record their maximum difference as well.
      assert np.array_equal(logits.argmax(-1).cpu().numpy(),ref['logits'].argmax(-1)),(cp,precision,T)
      checks.append(dict(T=T,predictions_reproduced=True,max_logit_difference=float(np.abs(logits.cpu().numpy()-ref['logits']).max())))
    report['case_groups']+=1;save(out/'status.json',report)
  rows=[]
  for name,parts in collected.items():
   arrays={k:np.concatenate([p[k] for p in parts],0) for k in parts[0]}
   np.savez_compressed(out/f'{name}.npz',**arrays)
   logits=arrays['logits'];target=arrays['target'];pred=logits.argmax(-1);meta=case_meta[name];digit=meta['digit']-1
   true=np.take_along_axis(logits,target[...,None],-1)[...,0];rivals=logits.copy()
   np.put_along_axis(rivals,target[...,None],-np.inf,-1);margin=true-rivals.max(-1)
   next_margin=true[:,digit]-logits[np.arange(len(target)),digit,target[:,digit+1]] if digit<9 else None
   record={**meta,'n':len(target),'string_correct':int((pred==target).all(-1).sum()),'token_correct':int((pred==target).sum()),
      'digit_correct':int((pred[:,digit]==target[:,digit]).sum()),'prediction':pred.tolist(),'target':target.tolist(),
      'margin':margin.tolist(),'target_vs_next_margin':next_margin.tolist() if next_margin is not None else None}
   save(out/f'{name}.json',record);rows.append(dict(name=name,**{k:v for k,v in record.items() if k not in ('prediction','target','margin','target_vs_next_margin')}))
  assert sha(path/'model.safetensors')==digest
  report.update(state='complete',finished=now(),elapsed_seconds=time.monotonic()-started,
                checks=checks,case_count=len(rows),prefix_immutable=True,payload_KV_equal_across_horizons=True,
                observation_and_sham_identity=True,earlier_answers_unchanged=True,weight_hash_unchanged=True)
  save(out/'summary.json',dict(cases=rows,checks=checks));save(out/'status.json',report)
  print('Completed',job['name'],len(rows),'cases in',round(report['elapsed_seconds'],1),'s',flush=True)
 except BaseException as exc:
  report.update(state='stopped',finished=now(),error=str(exc));save(out/'status.json',report);raise
 finally:probe.close()

if __name__=='__main__':main()
