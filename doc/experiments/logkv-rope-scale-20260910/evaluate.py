"""Unchanged standard evaluator plus preregistered paired boundary probes."""
import argparse
import importlib.util
import shutil
import sys
import torch
from common import SOURCE, ROOT, PAIRED_TS, name, run_dir, save, MODES
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(SOURCE/'exp/copying'))
spec=importlib.util.spec_from_file_location('standard_evaluate',SOURCE/'exp/copying/evaluate.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--checkpoint',choices=['best','final'],required=True); a=p.parse_args()
    out=ROOT/a.mode; records={}; current={}
    original_score=ev.score_logits; original_horizon=ev.eval_horizon
    def score(logits,labels):
        result=original_score(logits,labels)
        logits=logits[:,-10:].float(); target=labels[:,-10:]
        correct=logits.gather(-1,target[...,None]).squeeze(-1)
        rivals=logits.clone(); rivals.scatter_(-1,target[...,None],float('-inf'))
        records[str(current['T'])].append(dict(target=target.cpu().tolist(),
            prediction=logits.argmax(-1).cpu().tolist(), margin=(correct-rivals.max(-1).values).cpu().tolist()))
        return result
    def horizon(model,T,*args,**kwargs):
        current['T']=T; records[str(T)]=[]
        return original_horizon(model,T,*args,**kwargs)
    ev.score_logits=score; ev.eval_horizon=horizon
    sys.argv=[str(SOURCE/'exp/copying/evaluate.py'),'--run-name',name(a.mode),'--samples','256',
        '--max-t-exp','17','--seed','12345','--precision','bf16','--checkpoint',a.checkpoint,'--device','0']
    ev.main()
    save(out/f'digits_{a.checkpoint}.json',records)
    for file in ['results.json','plot.png']:
        src=run_dir(a.mode)/file; shutil.copy2(src,out/f'{src.stem}_{a.checkpoint}{src.suffix}')
    records={}; results={}
    folder='model_best' if a.checkpoint=='best' else 'model'
    model=ev.LogKVLM.from_pretrained(run_dir(a.mode)/folder).to(device='cuda',dtype=torch.bfloat16).eval()
    assert model.config.retrieval_rope_scale==MODES[a.mode]
    for T in PAIRED_TS:
        tok,st=horizon(model,T,32,torch.Generator().manual_seed(20260911),torch.device('cuda'),True)
        results[str(T)]=dict(token_acc=tok,string_acc=st,n=32)
        print(f'paired T={T} token={tok} string={st}',flush=True)
    save(out/f'paired_{a.checkpoint}.json',dict(seed=20260911,results=results,records=records))
if __name__=='__main__': main()
