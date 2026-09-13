"""Paired memories, answer-only attention traces, and IEEE fp32 replay.

Instrumentation observes inputs to the frozen attention function and returns
its unmodified output. 'memory overlap' describes slot spans, not recoverable
memory information or a causal importance score.
"""
import argparse
import json
import sys
import types
import numpy as np
import torch
from common import SOURCE, ROOT, BASELINE, RUN, CONTROL, save, sha
sys.path.insert(0,str(SOURCE))
from logkv import _local_rope, LogKV
from logkv_lm import LogKVLM
from evaluate_detail import digit_record, ev

PAIRED_TS=(3,16,8192,32768,49151,49152,49153,65536,131072)
FP32_TS=(49152,131072)


def distribution(attn,q,n,scale,flat,positions):
    """Materialize answer logits only; same local key rotation as production."""
    ks,vs=flat[:n],flat[n:2*n]
    loc,invalid=flat[2*n:3*n],flat[3*n:4*n]
    scores=[];values=[];spans=[];counts=[]
    C=attn.chunk_size
    assert attn.level_decay is None and not attn.relative_position_bias
    assert attn.relative_key is None and attn.retrieval_rope
    for i in range(n):
        k=ks[i][:,loc[i],:];v=vs[i][:,loc[i],:]
        j=(~invalid[i]).sum(-1)
        c=torch.arange(C-1,device=q.device)
        k=_local_rope(k,c[None]-j[:,None])
        score=torch.einsum('bld,blcd->blc',q,k)*scale-i*attn.level_decay_scale*np.log(C)
        scores.append(score.masked_fill(invalid[i][None],float('-inf')))
        values.append(v);counts.append(C-1)
        start=((positions//(C**(i+1)))[:,None]*C+c[None])*(C**i)
        spans.append((start<10)&(start+C**i>0)&~invalid[i])
    if flat[4*n:]:
        k,v=flat[4*n:]
        scores.append(((q*k).sum(-1)*scale)[...,None]);values.append(v[:,:,None])
        counts.append(1);spans.append(torch.zeros(len(positions),1,dtype=torch.bool,device=q.device))
    logits=torch.cat(scores,-1)
    # The diagnostic only records answer queries, which have past slots.
    probabilities=logits.float().softmax(-1) if logits.dtype!=torch.float64 else logits.softmax(-1)
    level_mass=torch.stack([x.sum(-1) for x in probabilities.split(counts,-1)],-1)
    overlap=(probabilities*torch.cat(spans,-1)[None]).sum(-1)
    return probabilities,torch.cat(values,2),level_mass,overlap


class Recorder:
    def __init__(self,model):
        self.offset=0;self.answer_start=0;self.traces={};self.originals=[]
        for layer,block in enumerate(model.layers):
            attn=block.attention;original=attn._attend_levels
            self.originals.append((attn,original))
            def wrapper(module,q,n,scale,*flat,_orig=original,_layer=layer):
                out=_orig(q,n,scale,*flat)
                start=max(0,self.answer_start-self.offset)
                if start<q.shape[1]:
                    sliced=(*flat[:2*n],*(x[start:] for x in flat[2*n:4*n]),
                            *(x[:,start:] for x in flat[4*n:]))
                    positions=torch.arange(self.offset+start,self.offset+q.shape[1],device=q.device)
                    probs,values,mass,overlap=distribution(module,q[:,start:],n,scale,sliced,positions)
                    B=q.shape[0]//module.num_heads;H=module.num_heads;L=q.shape[1]-start
                    record=dict(level_mass=mass.reshape(B,H,L,-1).float().cpu().numpy(),
                                memory_overlap_mass=overlap.reshape(B,H,L).float().cpu().numpy(),
                                self_mass=(mass[...,-1] if module.self_slot else torch.zeros_like(overlap)).reshape(B,H,L).float().cpu().numpy())
                    self.traces.setdefault(_layer,[]).append(record)
                return out
            attn._attend_levels=types.MethodType(wrapper,attn)
    def close(self):
        for attn,original in self.originals:attn._attend_levels=original


@torch.no_grad()
def infer(model,memory,T,precision,batch_limit=16,trace=True):
    device=next(model.parameters()).device
    all_logits=[];traces={};recorder=Recorder(model) if trace else None
    try:
        for start in range(0,len(memory),batch_limit):
            mem=memory[start:start+batch_limit];L=T+20
            ids=torch.zeros(len(mem),L,dtype=torch.long,device=device)
            ids[:,:10]=mem.to(device);ids[:,T+9:]=9
            hidden=None;answers=[]
            if recorder:recorder.answer_start=L-10;recorder.traces={}
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16' and device.type=='cuda'):
                for pos in range(0,L,8192):
                    if recorder:recorder.offset=pos
                    logits,hidden=model.step(ids[:,pos:pos+8192],hidden)
                    first=max(0,L-10-pos)
                    if first<logits.shape[1]:answers.append(logits[:,first:])
            all_logits.append(torch.cat(answers,1).float().cpu())
            if recorder:
                for layer,parts in recorder.traces.items():
                    for metric in ('level_mass','memory_overlap_mass','self_mass'):
                        key=f'layer{layer}_{metric}'
                        # Pad a newly created hierarchy level if an answer spans a boundary.
                        arrays=[r[metric] for r in parts]
                        if metric=='level_mass':
                            width=max(x.shape[-1] for x in arrays)
                            padded=[]
                            for x in arrays:
                                if x.shape[-1]<width:
                                    # Self is always the final column, distinct from hierarchy levels.
                                    if model.config.self_slot:
                                        x=np.concatenate((x[...,:-1],np.zeros((*x.shape[:-1],width-x.shape[-1]),dtype=x.dtype),x[...,-1:]),-1)
                                    else:x=np.pad(x,((0,0),(0,0),(0,0),(0,width-x.shape[-1])))
                                padded.append(x)
                            arrays=padded
                        traces.setdefault(key,[]).append(np.concatenate(arrays,axis=2))
    finally:
        if recorder:recorder.close()
    return torch.cat(all_logits),{k:np.concatenate(v,axis=0) for k,v in traces.items()}


def save_case(folder,name,model,memory,T,precision,batch_limit=16,metadata=None):
    logits,traces=infer(model,memory,T,precision,batch_limit)
    data=digit_record(logits,memory)
    correct=logits.argmax(-1)==memory
    data.update(T=T,precision=precision,n=len(memory),string_correct=int(correct.all(-1).sum()),
                token_correct=int(correct.sum()),errors_per_digit=(~correct).sum(0).tolist(),metadata=metadata or {})
    groups={}
    for group,mask in [('all',torch.ones(len(memory),dtype=torch.bool)),('correct',correct.all(-1)),('incorrect',~correct.all(-1))]:
        if not mask.any():continue
        groups[group]={k:v[mask.numpy()].mean(axis=(0,1)).tolist() for k,v in traces.items()}
    data['attention_mean_over_examples_and_heads']=groups
    save(folder/f'{name}.json',data)
    np.savez_compressed(folder/f'{name}.npz',logits=logits.numpy(),target=memory.numpy(),**traces)
    print(name,'string',data['string_correct'],'/',len(memory),'token',data['token_correct'],flush=True)
    return data


def validate():
    """Independent checks of trace meaning and instrumentation non-interference."""
    torch.set_num_threads(1)
    from configuration_logkv import LogKVConfig
    for self_slot in (False,True):
        torch.manual_seed(73)
        m=LogKVLM(LogKVConfig(vocab_size=10,d_model=16,num_heads=2,d_ff=32,num_layers=2,
             chunk_size=4,phase_emb=False,retrieval_rope=True,self_slot=self_slot,gated_attention=True)).double().eval()
        memory=torch.randint(1,9,(2,10))
        a,_=infer(m,memory,3,'fp32',trace=False)
        b,traces=infer(m,memory,3,'fp32')
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for key,val in traces.items():
            assert np.isfinite(val).all() and (val>=-1e-7).all() and (val<=1+1e-6).all()
            if key.endswith('level_mass'):np.testing.assert_allclose(val.sum(-1),1,atol=1e-6)
            if not self_slot and key.endswith('self_mass'):assert (val==0).all()
        # Construct a query at absolute position 7; level 0 spans 4,5,6,
        # level 1 spans 0..3. All past slots overlap the first ten tokens.
        attn=m.layers[0].attention;H=2;D=8
        q=torch.randn(H,1,D,dtype=torch.float64)
        ks=[torch.randn(H,3,D,dtype=torch.float64),torch.randn(H,1,D,dtype=torch.float64)]
        vs=[torch.randn_like(k) for k in ks]
        loc=[torch.tensor([[0,1,2]]),torch.tensor([[0,0,0]])]
        invalid=[torch.tensor([[False,False,False]]),torch.tensor([[False,True,True]])]
        flat=(*ks,*vs,*loc,*invalid)
        if self_slot:flat+=tuple(torch.randn_like(q) for _ in range(2))
        p,v,mass,overlap=distribution(attn,q,2,D**-.5,flat,torch.tensor([7]))
        expected=torch.einsum('blc,blcd->bld',p,v)
        actual=attn._attend_levels(q,2,D**-.5,*flat)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(overlap,1-mass[...,-1] if self_slot else torch.ones_like(overlap),atol=1e-12,rtol=1e-12)
    save(ROOT/'diagnostic_preflight.json',dict(instrumentation_preserves_logits=True,
          attention_distribution_matches_online_softmax=True,memory_span_classification_passed=True))
    print('Diagnostic preflight passed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    if args.validate_only:validate();return
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'diagnostics';folder.mkdir(exist_ok=False)
    fixed=torch.randint(1,9,(16,10),generator=torch.Generator().manual_seed(4321))
    # Reconstruct exactly the standard evaluation's T131072 memories. randint
    # draws depend only on the digit count here, not evaluation batch splits.
    gen=torch.Generator().manual_seed(12345)
    for T in ev.build_t_grid(17):
        batch=max(1,min(256,ev.TOKEN_BUDGET//(T+20)))
        standard=torch.cat([torch.randint(1,9,(min(batch,256-i),10),generator=gen)
                            for i in range(0,256,batch)])
    assert standard.shape==(256,10)
    cases=[]
    for mode,path in [('self-on',BASELINE),('self-off',RUN)]:
        for cp,sub in [('best','model_best'),('final','model')]:
            model_path=path/sub;weight_hash=sha(model_path/'model.safetensors')
            errors=None
            for precision in ('bf16','fp32'):
                dtype=torch.bfloat16 if precision=='bf16' else torch.float32
                model=LogKVLM.from_pretrained(model_path).to(device='cuda:0',dtype=dtype).eval()
                assert model.config.self_slot==(mode=='self-on') and not model.config.phase_emb
                assert model.config.retrieval_rope and not model.config.compressor_rope
                # Standard bf16 evaluation uses 'high'; keep its arithmetic for
                # reproduction, and explicitly disable TF32 for the fp32 probe.
                torch.set_float32_matmul_precision('high' if precision=='bf16' else 'highest')
                if precision=='fp32':torch.backends.cuda.matmul.allow_tf32=False
                for T in PAIRED_TS if precision=='bf16' else FP32_TS:
                    name=f'{mode}-{cp}-{precision}-paired-T{T}'
                    data=save_case(folder,name,model,fixed,T,precision,metadata=dict(memory_seed=4321,weight_sha256=weight_hash))
                    cases.append(dict(name=name,mode=mode,checkpoint=cp,kind='paired',T=T,precision=precision,
                                      n=data['n'],string_correct=data['string_correct'],token_correct=data['token_correct']))
                if mode=='self-on' and precision=='bf16':
                    data=save_case(folder,f'{mode}-{cp}-bf16-standard-T131072',model,standard,131072,precision,
                                   batch_limit=3,metadata=dict(evaluation_seed=12345,weight_sha256=weight_hash))
                    ref=json.loads((CONTROL/f'extension-131072/results_{cp}.json').read_text())['results']['131072']
                    assert data['string_correct']==round(ref['string_acc']*256)
                    assert data['token_correct']==round(ref['token_acc']*2560)
                    errors=torch.tensor(data['prediction']).ne(standard).any(-1)
                    errors=standard[errors]
                    assert len(errors)==9
                if mode=='self-on' and precision=='fp32' and errors is not None:
                    save_case(folder,f'{mode}-{cp}-fp32-bf16-errors-T131072',model,errors,131072,precision,
                              batch_limit=3,metadata=dict(selected_on='bf16 errors; not unbiased accuracy estimate',weight_sha256=weight_hash))
                del model;torch.cuda.empty_cache()
            assert sha(model_path/'model.safetensors')==weight_hash
    save(folder/'summary.json',dict(cases=cases,paired_memory_seed=4321,paired_samples=16,
         tf32_disabled_for_fp32=True,baseline_T131072_reproduced=True,
         checkpoint_hashes_unchanged=True,attention_note='Answer-query slot-span overlap, not causal memory importance'))

if __name__=='__main__': main()
