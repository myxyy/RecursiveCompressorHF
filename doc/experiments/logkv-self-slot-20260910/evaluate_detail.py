"""Frozen standard evaluation with per-example digit predictions and margins."""
import argparse
import importlib.util
import json
import sys
import torch
from common import SOURCE, ROOT, NAME, save
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(SOURCE/'exp/copying'))
spec=importlib.util.spec_from_file_location('frozen_evaluate',SOURCE/'exp/copying/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)


def digit_record(logits, target):
    logits=logits.float();target=target.long()
    correct=logits.gather(-1,target[...,None]).squeeze(-1)
    rivals=logits.clone();rivals.scatter_(-1,target[...,None],float('-inf'))
    return dict(target=target.cpu().tolist(),prediction=logits.argmax(-1).cpu().tolist(),
                margin=(correct-rivals.max(-1).values).cpu().tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',choices=['best','final'],required=True)
    args=p.parse_args(); records={};current={}
    original_score=ev.score_logits;original_horizon=ev.eval_horizon
    def score(logits,labels):
        result=original_score(logits,labels)
        records[str(current['T'])].append(digit_record(logits[:,-10:],labels[:,-10:]))
        return result
    def horizon(model,T,*args,**kwargs):
        current['T']=T;records[str(T)]=[]
        return original_horizon(model,T,*args,**kwargs)
    ev.score_logits=score;ev.eval_horizon=horizon
    sys.argv=[str(SOURCE/'exp/copying/evaluate.py'),'--run-name',NAME,'--samples','256',
              '--max-t-exp','17','--seed','12345','--precision','bf16','--checkpoint',args.checkpoint,'--device','0']
    ev.main()
    save(ROOT/f'digits_{args.checkpoint}.json',dict(checkpoint=args.checkpoint,seed=12345,records=records))

if __name__=='__main__': main()
