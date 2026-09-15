"""Inspect master/evaluation slopes and verify the full per-head training history."""
import json
import math
import torch
from safetensors.torch import load_file
from common import HERE, ROOT, MODES, run_dir, save

def main():
    rows=[]
    for mode in MODES:
        folder=run_dir(mode)
        history=[json.loads(line) for line in (folder/'beta_log.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==[0]+list(range(100,50001,100))
        for r in history:
            assert len(r['beta'])==2 and all(len(v)==8 for v in r['beta'])
            assert all(math.isfinite(v) for vs in r['beta'] for v in vs)
            assert r['alpha']==[[-v/math.log(4) for v in vs] for vs in r['beta']]
        expected=torch.full((2,8),math.log(4)).tolist()
        assert history[0]['beta']==expected
        best=json.loads((folder/'best.json').read_text())['step']
        for cp,sub,step in [('best','model_best',best),('final','model',50000)]:
            state=load_file(folder/sub/'model.safetensors')
            slopes=torch.stack([state[f'layers.{i}.attention.level_decay'] for i in range(2)]).float()
            assert slopes.tolist()==next(r['beta'] for r in history if r['step']==step)
            evaluated=slopes.bfloat16().float()
            rows.append(dict(task=mode,checkpoint=cp,step=step,beta=slopes.tolist(),
                beta_bf16_evaluation=evaluated.tolist(),alpha=(-slopes/math.log(4)).tolist(),
                min_beta=float(slopes.min()),max_beta=float(slopes.max()),
                negative_beta_heads=int((slopes<0).sum()),zero_beta_heads=int((slopes==0).sum()),
                weaker_decay_heads=int(((slopes>0)&(slopes<math.log(4))).sum())))
    save(HERE/'results/coefficients.json',dict(passed=True,rows=rows,
        convention='logit -= beta[layer,head] * level; beta<0 amplifies. alpha=-beta/log(C).',
        caveat='Main checkpoint stores fp32 coefficients; standard evaluation casts the entire model including coefficients to bf16.'))

if __name__=='__main__':main()
